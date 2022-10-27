package experiments;

import java.io.File;
import java.io.IOException;

public class InterruptProcessTest
{
	public static void main(String[] args) throws IOException, InterruptedException {
		String cmd = MpcbenchrunnerEvaluator.PYTHON_BINARY;
		cmd += " interrupted_print.py";

		ProcessBuilder pb = new ProcessBuilder(MpcbenchrunnerEvaluator.PYTHON_BINARY, "interrupted_print.py");
		pb.environment().put("PYTHONUNBUFFERED", "1");
		pb.redirectErrorStream(true);
		pb.inheritIO();
		Process p = pb.start();
		Thread.sleep(1000);
		p.destroy();
		Thread.sleep(1000);
		System.out.println("Finished the process");
	}
}
