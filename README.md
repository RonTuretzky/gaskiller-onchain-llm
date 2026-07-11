# Gas Killer · On-chain LLM — chat + bisection proof lab

Static frontend for the on-chain LLM running on the Gas Killer testnet (Sepolia):

- **Chat**: prompt the stories260K model (weights fully on-chain) or Qwen3-0.6B
  (overlay weights pinned by a 32-byte manifest). Tasks go to the Gas Killer router;
  the EigenLayer operator quorum simulates the billion-gas transformer off-chain,
  BLS-signs the single-slot diff, and the answer lands on Sepolia via verifyAndUpdate.
  The page reads answers straight from chain events — no backend.
- **Bisection Proof Lab**: interactive simulation of disputing a signed inference —
  per-token commitment checkpoints, log2(n) bisection to the first divergent segment,
  and single-segment re-execution (the SP1 slashing guest's job), with the real
  measured gas numbers.

Contracts: [solidity-sdk#56](https://github.com/gas-killer/solidity-sdk/pull/56) ·
service: [service#319](https://github.com/gas-killer/service/pull/319) ·
overlays: [gas-analyzer#168](https://github.com/gas-killer/gas-analyzer/pull/168)
