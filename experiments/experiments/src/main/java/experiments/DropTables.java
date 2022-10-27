package experiments;

import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import org.aeonbits.owner.ConfigFactory;

import java.io.File;

public class DropTables {
	public static void main(String[] args) throws ExperimentDBInteractionFailedException {
		IDatabaseConfig dbConfig = (IDatabaseConfig) ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(new File("runtime_configs/db.properties"));

		ExperimenterMySQLHandle handle = new ExperimenterMySQLHandle(dbConfig);
		handle.deleteDatabase();
	}
}
