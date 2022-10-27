package experiments.constraints;

import java.util.*;
import java.util.function.Predicate;

public class MPSPDZ_TCConstraints extends MPSPDZConstraints implements Predicate<List<String>> {

	@Override public boolean test(List<String> strings) {
		MPSPDZConstraints.KeyFields.PARAMS = new MPSPDZConstraints.Params(strings);
		if(!testProtocol(true)) {
			return false;
		}
		if(!NetConst.TEST_STD_CONSTRAINTS(KeyFields.tc_delay.intValueOf(), KeyFields.tc_rate.intValueOf())) {
			return false;
		}
		if(NetConst.TEST_DISABLED_TC(KeyFields.tc_delay.intValueOf(), KeyFields.tc_rate.intValueOf())) {
			if(KeyFields.tc_delay.valueOf().isPresent() && KeyFields.tc_rate.valueOf().isPresent()) {
				return false;
			}
		}
		return true;
	}
}

