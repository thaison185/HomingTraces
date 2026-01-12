import HomingTraces

TFSMs = list()
EFSMs = list()

# ==================================================================
# Add TFSM and EFSM models to be tested (Uncomment to include)
# ==================================================================

TFSMs.append(("TFSM example", HomingTraces.FSMWithTimeout("./tfsm_test.json")))
# TFSMs.append(("POP3", HomingTraces.FSMWithTimeout("./pop3_tfsm.json")))
# TFSMs.append(("SMTP", HomingTraces.FSMWithTimeout("./smtp.json")))
# TFSMs.append(("TCP", HomingTraces.FSMWithTimeout("./tcp_tfsm.json")))

# EFSMs.append(("Simple Authenticate Protocol - initial_state = DISCONNECTED", HomingTraces.EFSM("./simple_auth_efsm.json", initial_state="DISCONNECTED")))
# EFSMs.append(("Simple Authenticate Protocol", HomingTraces.FSMWithTimeout("./simple_auth_efsm.json")))
# EFSMs.append(("POP3 Protocol", HomingTraces.EFSM("./pop3_efsm.json")))


#=================================================================
# Set maximum length l
#=================================================================
l = 3

#=================================================================
# Generate homing traces for each TFSM and EFSM model
#=================================================================
for name, tfsm in TFSMs:
    print("\n" + "="*80)
    print(f"Processing TFSM: {name}")
    print("="*80)
    homing_traces = tfsm.io_homing_sequences(max_length=l)

    print(f"\nI/O Homing traces from TFSM model (total={len(homing_traces)}):")
    for trace, state in homing_traces:
        print(f"homing trace={trace} leads to State={state}")

    fsm = tfsm.create_fsm_without_timeout()
    homing_traces_fsm = tfsm.io_homing_sequences_from_model(fsm, max_length=l)

    print(f"\nI/O Homing traces from FSM model (total={len(homing_traces_fsm)}):")
    for trace, state in homing_traces_fsm:
        print(f"homing trace={trace} leads to State={state}")

    print(f"SUMMARY: {len(homing_traces)} homing traces from TFSM model, {len(homing_traces_fsm)} from FSM model")

for name, efsm in EFSMs:
    print("\n" + "="*50)
    print(f"Processing EFSM: {name}")
    print("="*80)
    nodes, edges = efsm.build_l_equivalent(l=l)
    homing_traces = efsm.generate_homing_traces(nodes, edges, l=l)

    print(f"\nHoming traces with initial_state = {efsm.initial_state} (total={len(homing_traces)}):")
    for t in homing_traces:
        print(f"len={t['length']} homing trace={t['trace']} leads to State={t['state']}")

