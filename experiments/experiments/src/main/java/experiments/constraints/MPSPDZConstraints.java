package experiments.constraints;

import java.security.Key;
import java.util.*;
import java.util.function.Predicate;

public class MPSPDZConstraints  implements Predicate<List<String>> {
	final static Map<String, Integer> PROTO_PARTY_NUM = Map.ofEntries(
			new AbstractMap.SimpleEntry<String, Integer>("mascot-party.x", -1),
			new AbstractMap.SimpleEntry<String, Integer>("mama-party.x", -1),
			new AbstractMap.SimpleEntry<String, Integer>("spdz2k-party.x", -1),
			new AbstractMap.SimpleEntry<String, Integer>("semi-party.x", -1),
			new AbstractMap.SimpleEntry<String, Integer>("semi2k-party.x", -1),
			new AbstractMap.SimpleEntry<String, Integer>("lowgear-party.x", -1),
			new AbstractMap.SimpleEntry<String, Integer>("highgear-party.x", -1),
			new AbstractMap.SimpleEntry<String, Integer>("cowgear-party.x", -1),
			new AbstractMap.SimpleEntry<String, Integer>("chaigear-party.x", -1),
			new AbstractMap.SimpleEntry<String, Integer>("hemi-party.x", -1),
			new AbstractMap.SimpleEntry<String, Integer>("temi-party.x", -1),
			new AbstractMap.SimpleEntry<String, Integer>("soho-party.x", -1),
			new AbstractMap.SimpleEntry<String, Integer>("semi-bin-party.x", -1),
			new AbstractMap.SimpleEntry<String, Integer>("tiny-party.x", -1),
			new AbstractMap.SimpleEntry<String, Integer>("tinier-party.x", -1),
			new AbstractMap.SimpleEntry<String, Integer>("replicated-ring-party.x", 3),
			new AbstractMap.SimpleEntry<String, Integer>("brain-party.x", 3),
			new AbstractMap.SimpleEntry<String, Integer>("ps-rep-ring-party.x", 3),
			new AbstractMap.SimpleEntry<String, Integer>("malicious-rep-ring-party.x", 3),
			new AbstractMap.SimpleEntry<String, Integer>("sy-rep-ring-party.x", 3),
			new AbstractMap.SimpleEntry<String, Integer>("rep4-ring-party.x", 4),
			new AbstractMap.SimpleEntry<String, Integer>("replicated-bin-party.x", 3),
			new AbstractMap.SimpleEntry<String, Integer>("malicious-rep-bin-party.x", 3),
			new AbstractMap.SimpleEntry<String, Integer>("ps-rep-bin-party.x", 3),
			new AbstractMap.SimpleEntry<String, Integer>("replicated-field-party.x", 3),
			new AbstractMap.SimpleEntry<String, Integer>("ps-rep-field-party.x", 3),
			new AbstractMap.SimpleEntry<String, Integer>("sy-rep-field-party.x", 3),
			new AbstractMap.SimpleEntry<String, Integer>("malicious-rep-field-party.x", 3),
			new AbstractMap.SimpleEntry<String, Integer>("atlas-party.x", -3),
			new AbstractMap.SimpleEntry<String, Integer>("shamir-party.x", -3),
			new AbstractMap.SimpleEntry<String, Integer>("malicious-shamir-party.x", -3),
			new AbstractMap.SimpleEntry<String, Integer>("sy-shamir-party.x", -3),
			new AbstractMap.SimpleEntry<String, Integer>("ccd-party.x", -3),
			new AbstractMap.SimpleEntry<String, Integer>("malicious-cdd-party.x", -3),
			new AbstractMap.SimpleEntry<String, Integer>("dealer-ring-party.x", -3),
			new AbstractMap.SimpleEntry<String, Integer>("yao-party.x", 2)
		);

	private final static String[] DISABLED_PROTOS = {
			// binary protocols
		"yao.x", "replicated-bin-party.x", "malicious-rep-bin-party.x", "ps-rep-bin-party.x", "temi-party.x", "semi-bin-party.x", "tiny-party.x", "tinier-party.x", "yao-party.x",
			// slow protocol
		"mama-party.x"
	};

	protected enum KeyFields {
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
	protected static class Params {

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

	protected boolean testProtocol(boolean selectMin) {

		if (KeyFields.protocol.valueOf().isEmpty()) {
			return true;
		}
		String protocolId = KeyFields.protocol.valueOf().get();
		if(Arrays.stream(DISABLED_PROTOS).anyMatch(protocolId::contains)) {
			return false;
		}
		if(!PROTO_PARTY_NUM.containsKey(protocolId)) {
			throw new IllegalArgumentException("Unknown protocol: " + protocolId);
		}
		if(KeyFields.world_size.intValueOf().isPresent()) {
			int worldSize = KeyFields.world_size.intValueOf().get();
			int partyNum = PROTO_PARTY_NUM.get(protocolId);
			if(partyNum == -1) {
				if(selectMin && worldSize != 2) {
					return false;
				}
				// pass
			}
			else if(partyNum == -3) {
				if(worldSize < 3)
					return false;
				if(selectMin && worldSize != 3) {
					return false;
				}
			}
			else if(partyNum != worldSize) {
				return false;
			}
		}
		return true;
	}

	@Override public boolean test(List<String> strings) {
		KeyFields.PARAMS = new Params(strings);
		if(!NetConst.TEST_DISABLED_TC(KeyFields.tc_delay.intValueOf(), KeyFields.tc_rate.intValueOf())) {
			return false;
		}
		if(!testProtocol(false)) {
			return false;
		}
		return true;
	}
}

