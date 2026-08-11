# Hallucination Propagation in Multi-Agent LLMs

## Project Structure
```
hallucination_project/
├── data/
│   └── hallucination_propagation_dataset.csv   # your 1000-row synthetic dataset
├── experiments/
│   ├── exp1_seeding_method.py                  # Experiment 1: seeding method vs propagation
│   ├── exp2_generation_decay.py                # Experiment 2: propagation decay across generations
│   ├── exp3_confidence_amplification.py        # Experiment 3: confidence amplification
│   └── exp4_propagation_vs_persistence.py      # Experiment 4: propagation vs persistence
├── results/
│   └── ()
├── utils/
│   ├── azure_client.py                         # Azure OpenAI client wrapper
│   └── prompts.py                              # All prompt templates
├── config.py                                   # API keys + model config
├── run_all.py                                  # Run all experiments in sequence
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
cp config.example.py config.py   # then fill in your Azure credentials
```

## Running Experiments
```bash
python experiments/exp1_seeding_method.py
python experiments/exp2_generation_decay.py
python run_all.py   # or run all at once
```
