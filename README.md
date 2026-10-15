# Rosatom 5D Additive Manufacturing MCP Server

MCP-сервер (ядро слайсинга, анализа геометрии и промышленной безопасности) для кейса Росатома по FDM 3D/5D-печати.

## Архитектура
- `mcp_server/mesh_analyzer.py` — глубокий анализ геометрии меша (габариты, объем, нависания > 45°, герметичность, адгезия, вердикт 3D/5D).
- `mcp_server/security_guard.py` — модуль физической безопасности (лимиты температур, скоростей, статический анализ G-кода, SHA-256).
- `mcp_server/slicer_cli.py` — запуск PrusaSlicer/OrcaSlicer CLI с математическим fallback-генератором G-кода.
- `mcp_server/five_axis_postproc.py` — пост-процессор для 5-осевой печати Stereotech (оси A/C).
- `mcp_server/profiles/filaments.json` — база промышленных филаментов (PLA, ABS, PETG, PA12-CF, PEEK).
- `mcp_server/server.py` — FastMCP сервер с зарегистрированными инструментами.
