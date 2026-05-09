# JTL Analyzer

Sistema multiagente para analizar resultados de pruebas JMeter (archivos .jtl) y
generar reportes en Excel. Soporta análisis individual de un JTL y comparación
entre múltiples JTLs (en hitos futuros).

## Estado actual: Hito 0

Núcleo mínimo funcional. Sólo el orquestador, el cargador y el estadístico
básico están implementados. El sistema acepta un único JTL en formato CSV y
produce métricas básicas (count, average, error rate) globales y por feature.

## Arquitectura

Sistema de agentes especializados coordinados por un agente planificador. Regla
fundamental:

- Los **agentes especialistas** (cargador, estadístico; en hitos futuros:
  tendencias, anomalías, errores, comparativo, visualizador, reportería) son
  **código Python determinístico**. No invocan LLMs. Producen dataclasses con
  números, listas y referencias temporales — nunca prosa.
- Sólo **dos roles usan LLM**: el orquestador (decide qué pasos ejecutar) y, en
  hitos futuros, el narrador (convierte hallazgos en prosa interpretativa).
- Toda interacción con LLMs pasa por la abstracción `LLMProvider`
  (`src/jtl_analyzer/providers/`). **No usar SDKs de proveedores directamente
  desde otros módulos.**

### Flujo del orquestador

1. Recibe el pedido del usuario más metadata (paths a JTLs, opciones).
2. Genera un `Plan` — lista de `PlanStep` con dependencias — invocando al LLM
   en modo JSON estructurado.
3. Valida el plan: cada step referencia un agente registrado, y las
   dependencias forman un DAG sin ciclos.
4. Ejecuta los steps en orden topológico, pasando los outputs de pasos previos
   como inputs según las dependencias declaradas.
5. Retorna el resultado final estructurado.

En Hito 0 hay un único intent soportado ("analizar un JTL"), que produce un
plan canónico de dos pasos: `loader` → `statistician`. La arquitectura está
preparada para más intents sin cambios estructurales.

### Abstracción de proveedor LLM

Todos los proveedores implementan `LLMProvider` (en `providers/base.py`):

```python
class LLMProvider(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse: ...
```

Para `json_mode=True`, la implementación más portable entre proveedores en
Hito 0 es prompt engineering: instruir al modelo a responder sólo con JSON
válido y parsear la respuesta. Los proveedores tienen APIs nativas para
salida estructurada (Anthropic con tool use, OpenAI con `response_format`,
Gemini con `response_mime_type`) que pueden incorporarse en hitos posteriores
si el prompt engineering muestra inestabilidad.

## Stack

- Python 3.13+
- pandas (manipulación de datos)
- pytest (tests)
- python-dotenv (carga de `.env`)
- SDKs oficiales de los proveedores LLM (`anthropic`; `openai` y
  `google-generativeai` en hitos futuros)
- pip + venv (gestión de entorno)

## Convenciones

- **Naming**: módulos y archivos en `snake_case`, clases en `PascalCase`,
  funciones y variables en `snake_case`.
- **Type hints**: obligatorios en todas las funciones públicas y métodos de
  clases. Usar sintaxis moderna (`list[str]` en vez de `List[str]`,
  `X | None` en vez de `Optional[X]`).
- **Modelos de datos**: usar `@dataclass(frozen=True)` para los reports y
  modelos inmutables. Reservar pydantic para casos donde haga falta validación
  desde JSON externo (parsing del plan que devuelve el LLM, por ejemplo).
- **Errores**: definir excepciones específicas en `core/exceptions.py`
  (`InvalidJTLError`, `PlanValidationError`, `ProviderError`, etc.). No
  retornar `None` o `False` para señalar fallas.
- **Logging**: usar el módulo `logging` estándar, configurado en `config.py`.
  No usar `print()` excepto en `cli.py`.
- **Docstrings**: estilo Google, obligatorias en clases públicas y en el API
  público de cada módulo.

## Comandos

```bash
# Setup inicial
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows
pip install -e ".[dev]"

# Configuración
cp .env.example .env
# Editar .env y poner las API keys

# Correr todos los tests
pytest

# Correr un test específico
pytest tests/test_loader.py -v

# Correr el CLI sobre un JTL de prueba
python -m jtl_analyzer.cli analyze tests/fixtures/small_clean.jtl
```

## Variables de entorno

Definidas en `.env` (no commitear). Ver `.env.example` para la plantilla.

- `LLM_PROVIDER`: `"anthropic"` | `"openai"` | `"gemini"` (default: `"anthropic"`)
- `LLM_MODEL`: identificador del modelo
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`: según el proveedor
  configurado

## Reglas importantes al contribuir

1. **Nunca** llamar al SDK de un proveedor LLM directamente fuera de
   `providers/`. Siempre pasar por `LLMProvider`.
2. **Nunca** invocar un LLM desde un agente especialista. Si una funcionalidad
   parece necesitar "interpretación", el especialista produce el dato
   estructurado y un agente con LLM (orquestador o futuro narrador) lo
   interpreta.
3. **Todo agente nuevo** debe (a) registrarse en el orquestador con un nombre
   único, (b) consumir y producir dataclasses definidas en `core/models.py`,
   y (c) tener tests unitarios.
4. **Tests primero** para los especialistas: son código puro, son
   trivialmente testeables. Los tests del orquestador deben mockear
   `LLMProvider` para no depender de llamadas externas.
5. **No commitear** archivos JTL grandes (`samples/` está en `.gitignore`),
   `.env`, `.venv/`, ni outputs generados (Excel finales).

## Próximos hitos

- **Hito 1**: implementación de `OpenAIProvider` y `GeminiProvider`. Agentes
  de tendencias, anomalías y errores. Ampliación del estadístico (percentiles,
  throughput).
- **Hito 2**: agente comparativo (multi-JTL) con tests de significancia.
- **Hito 3**: visualizador y reportería con template Excel embebido como
  recurso del paquete.
- **Hito 4**: agente narrador. Interfaz Streamlit para modo conversacional /
  granular.