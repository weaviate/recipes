# Phoenix Setup
import phoenix as px
phoenix_session = px.launch_app()

# Note this will also catch calls to Weaviate when connected to the DSPy WeaviateRM
from openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
span_otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=span_otlp_exporter))

trace_api.set_tracer_provider(tracer_provider=tracer_provider)
DSPyInstrumentor().instrument()
