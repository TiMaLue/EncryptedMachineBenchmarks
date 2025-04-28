package experiments;

import ai.libs.jaicore.experiments.Experiment;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentFailurePredictionException;
import com.google.gson.Gson;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class MpcbenchrunnerEvaluator implements IExperimentSetEvaluator {

  private static final Logger logger = LoggerFactory.getLogger(MpcbenchrunnerEvaluator.class);
  private static final long PROCESS_WAIT_TIME_SEC = (int) (7200 / 1);

  public static final String PYTHON_BINARY = "../mpcbenchrunner/bin/python";

  private File prepareInputConfig(Map<String, String> experimentParams) throws IOException {
    Gson gson = new Gson();
    final File inputFile;
    inputFile = File.createTempFile("input-", ".json");
    logger.info("Input config file: {}", inputFile);
    String inputConfigStr = gson.toJson(experimentParams);
    Files.writeString(Path.of(inputFile.getAbsolutePath()), inputConfigStr, Charset.defaultCharset());
    return inputFile;
  }

  private Thread redirectOut(Process p) {
    Thread t = new Thread(() -> {
      try {
        p.getInputStream().transferTo(System.out);
      } catch (IOException ignored) {
      }
    });
    t.start();
    return t;
  }

  private void startBenchmarkProcess(File inputFile, File outputFile)
      throws InterruptedException, ExperimentEvaluationFailedException {
    ProcessBuilder pb = new ProcessBuilder(PYTHON_BINARY, "-m", "mpcbenchrunner.runner",
        inputFile.getAbsolutePath(), outputFile.getAbsolutePath());
    pb.redirectErrorStream(true);
    // pb.inheritIO();
    pb.environment().put("MPCBR_SETTINGS_LOG_CONFIG_LOCATION", "runtime_configs/mpcbenchrunner_logconfig.yaml");
    pb.environment().put("MPCBR_SETTINGS_BENCHMARK_DATA_DIR", "/bench_data");
    pb.environment().put("PYTHONUNBUFFERED", "1");
    logger.info("Running benchmark: {}", String.join(" ", pb.command()));
    logger.info("========================================");
    logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    Process p = null;
    Thread t = null;
    try {
      p = pb.start();
      t = redirectOut(p);
      p.waitFor(PROCESS_WAIT_TIME_SEC, TimeUnit.SECONDS);
      if (p.isAlive()) {
        throw new ExperimentEvaluationFailedException("Timeout");
      }
      if (p.exitValue() != 0) {
        throw new ExperimentEvaluationFailedException("Exit value: " + p.exitValue());
      }
    } catch (IOException e) {
      throw new ExperimentEvaluationFailedException(e);
    } finally {
      if (t != null) {
        t.interrupt();
      }
      if (p != null && p.isAlive()) {
        try {
          logger.warn("Sending kill signal to process.");
          Runtime.getRuntime().exec("kill -SIGINT " + p.pid());
          Thread.sleep(5000);
        } catch (IOException e) {
          logger.warn("Stopping process failed: ", e);
        }
        if (p.isAlive())
          logger.warn("Process {} is still alive.", p.pid());
        p.destroy();
      }
      logger.info("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
      logger.info("========================================");
    }
  }

  private File prepareOutputMeasurements() throws IOException {
    return File.createTempFile("output-", ".json");
  }

  private Map<String, Object> readMeasurements(File outputFile) throws IOException {
    Gson gson = new Gson();
    try (FileInputStream fis = new FileInputStream(outputFile)) {
      BufferedInputStream bufferedInputStream = new BufferedInputStream(fis);
      InputStreamReader inputStreamReader = new InputStreamReader(bufferedInputStream);
      return gson.fromJson(inputStreamReader, Map.class);
    }
  }

  @Override
  public void evaluate(ExperimentDBEntry experimentEntry, IExperimentIntermediateResultProcessor processor)
      throws ExperimentEvaluationFailedException, ExperimentFailurePredictionException, InterruptedException {
    Experiment experiment = experimentEntry.getExperiment();
    Map<String, String> keyFields = experiment.getValuesOfKeyFields();
    final File inputFile;
    final File outputFile;
    try {
      inputFile = prepareInputConfig(keyFields);
      outputFile = prepareOutputMeasurements();
      inputFile.deleteOnExit();
      outputFile.deleteOnExit();
    } catch (IOException e) {
      throw new ExperimentEvaluationFailedException(e);
    }

    try {
      startBenchmarkProcess(inputFile, outputFile);
    } catch (ExperimentEvaluationFailedException e) {
      logger.info("Experiment failed: {}", e.getMessage());
      throw (e);
    }

    Map<String, Object> measurements;

    try {
      measurements = readMeasurements(outputFile);
    } catch (IOException e) {
      logger.error("Couldn't read results.");
      throw new ExperimentEvaluationFailedException("Reading results.", e);
    }
    processor.processResults(measurements);
  }
}
