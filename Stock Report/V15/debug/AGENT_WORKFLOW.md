# Multi-Agent Workflow Protocol

## Workflow Steps

### Step 1: Builder Writes Code
**Agent Role**: Builder (any agent 1-8)
**Action**: 
1. Check `AGENT_COORDINATION.md` for assigned tasks
2. Check `TASK_QUEUE.md` for specific tasks
3. Write code for assigned module
4. Follow coding standards
5. Update task status: `[ ]` → `[IN PROGRESS]`

**Output**: Code file(s) ready for review

### Step 2: Reviewer Critiques
**Agent Role**: Reviewer (different agent or dedicated reviewer)
**Action**:
1. Read the code
2. Check for:
   - Bugs and logic errors
   - Security flaws
   - Missing edge cases
   - Dependency issues
   - Performance problems
3. Create review document with findings
4. Mark: `[REVIEWED]` with findings

**Output**: Review document with issues found

### Step 3: Builder Updates Code
**Agent Role**: Original Builder
**Action**:
1. Read review findings
2. Fix all identified issues
3. Address edge cases
4. Update code
5. Mark: `[FIXED]`

**Output**: Updated code addressing all review issues

### Step 4: Tester Writes Tests
**Agent Role**: Testing Agent (Agent 7) or dedicated tester
**Action**:
1. Write pytest tests for the module
2. Test all functions
3. Test edge cases
4. Test error handling
5. Run tests and verify they pass
6. Mark: `[TESTED]`

**Output**: Test file with passing tests

### Step 5: Builder Addresses Test Issues
**Agent Role**: Original Builder
**Action**:
1. Review test failures
2. Fix any issues found by tests
3. Ensure all tests pass
4. Mark: `[COMPLETE]`

**Output**: Module complete with passing tests

## Agent Communication

### Format for Status Updates

```markdown
[STATUS UPDATE]
Agent: [Agent Number/Name]
Module: [module_name.py]
Status: [IN PROGRESS | REVIEWED | FIXED | TESTED | COMPLETE]
Notes: [Any relevant notes]
Timestamp: [ISO timestamp]
```

### Format for Review Requests

```markdown
[REVIEW REQUEST]
Agent: [Agent Number/Name]
Module: [module_name.py]
Ready for: [Code Review | Testing]
Notes: [Specific areas to focus on]
```

### Format for Handoff

```markdown
[HANDOFF]
From: [Agent Number/Name]
To: [Agent Number/Name]
Module: [module_name.py]
Status: [COMPLETE]
Next Steps: [What next agent should do]
```

## Current Active Agents

### Agent 8: Debug Agent
**Current Task**: Create debug utilities
**Status**: IN PROGRESS
**Files Working On**:
- `debug/risk_debugger.py` - Next
- `debug/browser_debugger.py`
- `debug/learning_debugger.py`
- `debug/integration_debugger.py`

**Blockers**: None
**Estimated Completion**: [Update when known]

## Coordination Rules

1. **One Module Per Agent**: Each agent works on one module at a time
2. **Update Status**: Always update status in coordination files
3. **No Overwrites**: Don't modify another agent's code without agreement
4. **Communication**: Use status update format for all communications
5. **Review Before Test**: Code must be reviewed before testing
6. **Test Before Complete**: Code must be tested before marking complete

## Example Workflow

### Example: Debug Agent Creating risk_debugger.py

**Step 1 - Builder (Agent 8)**:
```
[STATUS UPDATE]
Agent: 8 (Debug Agent)
Module: debug/risk_debugger.py
Status: IN PROGRESS
Notes: Creating risk calculation debugger
```

**Step 2 - Reviewer (Agent 3 - Risk Agent)**:
```
[REVIEW REQUEST]
Agent: 8 (Debug Agent)
Module: debug/risk_debugger.py
Ready for: Code Review

[REVIEW FINDINGS]
- Missing error handling for zero equity case
- ATR calculation doesn't handle NaN values
- Position sizing test missing edge cases
```

**Step 3 - Builder (Agent 8)**:
```
[STATUS UPDATE]
Agent: 8 (Debug Agent)
Module: debug/risk_debugger.py
Status: FIXED
Notes: Addressed all review findings
```

**Step 4 - Tester (Agent 7)**:
```
[STATUS UPDATE]
Agent: 7 (Testing Agent)
Module: debug/risk_debugger.py
Status: TESTED
Notes: All tests passing
```

**Step 5 - Builder (Agent 8)**:
```
[STATUS UPDATE]
Agent: 8 (Debug Agent)
Module: debug/risk_debugger.py
Status: COMPLETE
Notes: Ready for use
```

## Getting Started as New Agent

1. Read `AGENT_COORDINATION.md` to understand roles
2. Check `TASK_QUEUE.md` for available tasks
3. Pick an unassigned task
4. Update status: Mark as `[IN PROGRESS]`
5. Start coding following the workflow
6. Request review when ready
7. Fix issues found
8. Mark as `[COMPLETE]` when done

## Conflict Resolution

If two agents want the same module:
1. Check `AGENT_COORDINATION.md` for assignments
2. First agent to mark `[IN PROGRESS]` gets it
3. Other agent picks different task
4. If conflict, use task priority in `TASK_QUEUE.md`

## Quality Standards

All agents must ensure:
- ✅ Code follows V15 patterns
- ✅ Uses relative paths
- ✅ Has docstrings
- ✅ Has error handling
- ✅ Has tests
- ✅ No linter errors
- ✅ Follows plan specifications

