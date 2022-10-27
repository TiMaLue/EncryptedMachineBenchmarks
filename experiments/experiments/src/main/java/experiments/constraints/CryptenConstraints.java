package experiments.constraints;

import javax.swing.text.html.Option;
import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;

public class CryptenConstraints implements Predicate<List<String>> {
	private class Params {

		int scenario = 0;
		int target = 1;
		int world_size = 2;
		int plain_text = 3;
		int batch_size = 4;
		int ttp = 5;
		int tc_delay = 6;
		int tc_rate = 7;


		List<String> strings;

		Params(List<String> strings) {
			this.strings = strings;
		}
		Optional<String> valueOf(int paramIndex) {
			if(strings.size() <= paramIndex) {
				return Optional.empty();
			}
			return Optional.of(strings.get(paramIndex));
		}

		Optional<Boolean> booleanValueOf(int paramIndex) {
			return valueOf(paramIndex).map(pt -> pt.equals("1"));
		}
		Optional<Integer> intValueOf(int paramIndex) {
			return valueOf(paramIndex).map(Integer::parseInt);
		}
		private Optional<Boolean> isPlaintext() {
			return booleanValueOf(plain_text);
		}
		private Optional<Integer> worldSize() {
			return intValueOf(world_size);
		}
		private Optional<Integer> tcRate() {
			return intValueOf(tc_rate);
		}
		private Optional<Integer> tcDelay() {
			return intValueOf(tc_delay);
		}
		private Optional<Boolean> ttp() {
			return booleanValueOf(ttp);
		}
	}

	@Override public boolean test(List<String> strings) {
		Params p = new Params(strings);
		if(p.isPlaintext().isPresent()) {
			if(p.isPlaintext().get()) {
				if(p.worldSize().map(ws ->  ws > 1).orElse(false)) {
					return false;
				}
				if(p.ttp().orElse(false)) {
					return false;
				}
				// if(p.tcRate().map(tcRate -> tcRate != -1).orElse(false)) {
				// 	return false;
				// }
				// if(p.tcDelay().map(tcDelay -> tcDelay != -1).orElse(false)) {
				// 	return false;
				// }

				if(!NetConst.TEST_DISABLED_TC(p.tcDelay(), p.tcRate())) {
					return false;
				}
			} else {
				if(p.worldSize().map(ws -> ws < 2).orElse(false)) {
					return false;
				}
			}
		}
// 		if(p.tcDelay().isPresent() && p.tcRate().isPresent()) {
// 			if (p.tcDelay().get() >= 0 && p.tcRate().get() >= 0) {
// 				return false;
// 			}
// //			if (p.tcDelay().get().equals(5) && !p.tcRate().get().equals(10000)) {
// //				return false;
// //			}
// //			if (p.tcDelay().get().equals(15) && !p.tcRate().get().equals(50)) {
// //				return false;
// //			}
// 		}
		if(!NetConst.TEST_STD_CONSTRAINTS(p.tcDelay(), p.tcRate())) {
			return false;
		}
		return true;
	}
}
