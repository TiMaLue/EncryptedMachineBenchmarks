package experiments.constraints;

import java.util.*;
import java.util.function.Predicate;

public class TFEncryptedConstraints implements Predicate<List<String>> {

	final static Map<String, Integer> PROTO_PARTY_NUM = Map.ofEntries(
			new AbstractMap.SimpleEntry<String, Integer>("aby3", 3),
			new AbstractMap.SimpleEntry<String, Integer>("securenn", 3),
			new AbstractMap.SimpleEntry<String, Integer>("pond", 2)
		);

	private enum KeyFields {
		// keyfields = scenario:varchar(500),target:varchar(500),protocol:varchar(500),world_size:int,batch_size:int,tc_delay:int,tc_rate:int
		scenario,
		target,
		protocol,
		world_size,
		batch_size,
		tc_delay,
		tc_rate;
		static Params PARAMS;
		Optional<String> valueOf() {
			return PARAMS.valueOf(this);
		}
		Optional<Boolean> booleanValueOf() {
			return PARAMS.booleanValueOf(this.ordinal());
		}
		Optional<Integer> intValueOf() {
			return PARAMS.intValueOf(this.ordinal());
		}
	}
	private static class Params {

		private final List<String> strings;

		Params(List<String> strings) {
			this.strings = strings;
		}

		Optional<String> valueOf(int paramIndex) {
			if(strings.size() <= paramIndex) {
				return Optional.empty();
			}
			return Optional.of(strings.get(paramIndex));
		}
		Optional<String> valueOf(KeyFields kf) {
			return valueOf(kf.ordinal());
		}

		Optional<Boolean> booleanValueOf(int paramIndex) {
			return valueOf(paramIndex).map(pt -> pt.equals("1"));
		}
		Optional<Integer> intValueOf(int paramIndex) {
			return valueOf(paramIndex).map(Integer::parseInt);
		}
		Optional<Boolean> booleanValueOf(KeyFields kf) {
			return booleanValueOf(kf.ordinal());
		}
		Optional<Integer> intValueOf(KeyFields kf) {
			return intValueOf(kf.ordinal());
		}
	}

	private boolean testProtocolWorldSize() {

		if (KeyFields.protocol.valueOf().isEmpty()) {
			return true;
		}
		String protocolId = KeyFields.protocol.valueOf().get();
		if(KeyFields.world_size.intValueOf().isPresent()) {
			int worldSize = KeyFields.world_size.intValueOf().get();
			int partyNum = PROTO_PARTY_NUM.get(protocolId);
			if(partyNum != worldSize) {
				return false;
			}
		}
		return true;
	}

	@Override public boolean test(List<String> strings) {
		KeyFields.PARAMS = new Params(strings);
//		if(KeyFields.tc_delay.intValueOf().map(tcDelay -> tcDelay > 2).orElse(false)) {
//			return false;
//		}
//		if(KeyFields.tc_rate.intValueOf().map(tcRate -> tcRate < 1000).orElse(false)) {
//			return false;
//		}
		if(!testProtocolWorldSize()) {
			return false;
		}
		if(	KeyFields.scenario.valueOf().map(s -> !s.contains("image_cls")).orElse(false) &&
				KeyFields.batch_size.intValueOf().map(b -> b < 2).orElse(false)) {
			return false;
		}
		if(!NetConst.TEST_STD_CONSTRAINTS(KeyFields.tc_delay.intValueOf(), KeyFields.tc_rate.intValueOf())) {
			return false;
		}
		return true;
	}
}

