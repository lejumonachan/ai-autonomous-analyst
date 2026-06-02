import textwrap


def scientific_answer(question, context):
    answer = f"""
### Scientific AI Interpretation

Based on the uploaded experiment context, the system can support analysis of:

- Mission telemetry stability
- Biological experiment conditions
- Environmental risk factors
- Possible anomaly patterns
- Experiment outcome indicators

### Question
{question}

### Relevant Context Preview
{context[:1500]}

### Recommendation
Review mission health, anomaly detection, risk scoring, and prediction results together before making scientific conclusions.
"""
    return textwrap.dedent(answer)


def generate_scientific_summary(context):
    summary = f"""
## Helios AI Scientific Mission Report

### 1. Mission Overview
The uploaded experiment data/document was processed through the Helios AI platform for scientific mission analysis. The system reviewed available telemetry, biological indicators, experiment conditions, and operational signals.

### 2. Key Observations
- The mission dataset contains experiment records suitable for monitoring and analysis.
- Environmental and biological variables can be evaluated for stability.
- Mission health, anomaly detection, and risk scoring should be reviewed together.
- Prediction outputs can support experiment success or failure assessment.

### 3. Risk Assessment
Potential risks may include:
- Missing or inconsistent telemetry values
- Abnormal environmental conditions
- High variation in biological response signals
- Possible contamination or unstable growth conditions
- Mission outcome uncertainty

### 4. Scientific Recommendations
- Monitor radiation, temperature, oxygen, pH, and biological growth indicators carefully.
- Use anomaly detection before making operational decisions.
- Review mission risk classification before continuing sensitive biological experiments.
- Compare prediction results with domain knowledge and experimental baselines.
- Generate repeated reports after each experiment cycle.

### 5. AI Interpretation
Helios AI provides a decision-support layer for autonomous space biolab experiments by combining telemetry monitoring, anomaly detection, predictive modeling, and scientific reporting.

### 6. Context Preview
{context[:2500]}
"""
    return textwrap.dedent(summary)