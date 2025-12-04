PROJECT: kaiser_rag

Folder structure:

kaiser_rag/
├── backend/
│   └── src/
│       ├── agents/
│       │   ├── conversational_agent.py
│       │   ├── orchestrator_agent.py
│       │   ├── retrieve_agent.py
│       │   ├── router_agent.py
│       │   └── transactional_agent.py
│       ├── agent_workflow/
│       │   └── state.py
│       ├── core/
│       │   ├── config.py
│       │   └── logging.py
│       ├── models/
│       ├── tools/
│       │   ├── check_plan_coverage.py
│       │   ├── fetch_billing_info.py
│       │   ├── rag_search_tool.py
│       │   └── schedule_appointment.py
│       └── utils/
├── frontend/
├── myenv/
├── DESIGN.md
├── README.md
└── requirements.txt