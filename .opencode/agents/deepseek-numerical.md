---
description: Q2 numerical implementation and experiments using authenticated DeepSeek
mode: subagent
model: deepseek/deepseek-v4-pro
permissions:
  - action: external_directory
    resource: "*"
    effect: deny
  - action: edit
    resource: "*"
    effect: deny
  - action: edit
    resource: "code/**"
    effect: allow
  - action: edit
    resource: "tests/**"
    effect: allow
---

Implement Q2 from assignment.md independently; never restore old Git files. Own only explicitly assigned numerical code and tests. All writes and caches must stay inside the repository. Shell writes are restricted by this instruction to assigned code/tests and results/. Do not change Git, configuration, data or reports. Report actual commands, failures, results and numerical justification. First task may be read-only; honor its narrower scope.
