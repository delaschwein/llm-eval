
import json
import os
import glob
import pandas as pd
from plotnine import ggplot, aes, geom_bar, ggsave, theme, element_text, labs, facet_wrap
from sklearn.metrics import f1_score, precision_score, recall_score

def shorten_model_name(name):
    """Shortens long model names for display."""
    name = name.replace('_evaluation_results', '').replace('.json', '')
    name = name.replace('anthropic_claude-3.5-sonnet', 'Claude 3.5 Sonnet')
    name = name.replace('gpt-3.5-turbo', 'GPT-3.5 Turbo')
    name = name.replace('meta-llama_llama-3-70b-instruct', 'Llama3 70B')
    name = name.replace('meta-llama_llama-3-8b-instruct', 'Llama3 8B')
    name = name.replace('mistralai_mixtral-8x7b-instruct', 'Mixtral 8x7B')
    name = name.replace('openai_gpt-4', 'GPT-4')
    name = name.replace('openai_gpt-4o-mini', 'GPT-4o Mini')
    name = name.replace('prometheus', 'Prometheus')
    name = name.replace('ensemble_', 'Ens: ')
    name = name.replace('dafe_', 'DAFE: ')
    name = name.replace('_', '/')
    return name

def calculate_metrics(file_path):
    """Calculates F1, precision, and recall for each criterion."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    results = []
    is_ensemble = 'ensemble' in os.path.basename(file_path)
    is_dafe = 'dafe' in os.path.basename(file_path)

    for criterion in ['relevance', 'attributes', 'facts', 'preference']:
        predictions = []
        ground_truth = []
        for question, aspects in data.items():
            if criterion in aspects:
                values = aspects[criterion]
                if is_ensemble:
                    prediction = values.get('ensemble_acceptable')
                    human_annotation = values.get('ensemble_human_annotation')
                elif is_dafe:
                    prediction = values.get('acceptable')
                    human_annotation = values.get('human_annotation')
                else: # Single model
                    prediction = values.get('acceptable')
                    human_annotation = values.get('human_annotation')

                if prediction is not None and human_annotation is not None:
                    predictions.append(prediction)
                    ground_truth.append(human_annotation)
        
        if predictions:
            f1 = f1_score(ground_truth, predictions)
            precision = precision_score(ground_truth, predictions)
            recall = recall_score(ground_truth, predictions)
        else:
            f1, precision, recall = 0.0, 0.0, 0.0
        
        results.append({
            'criterion': criterion.capitalize(), 
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        })

    return results

def generate_f1_chart(df):
    """Generate the faceted F1 score comparison bar chart."""
    chart = (
        ggplot(df, aes(x='model', y='f1_score', fill='type'))
        + geom_bar(stat='identity')
        + facet_wrap('~criterion', ncol=2)
        + theme(axis_text_x=element_text(angle=60, hjust=1, size=8), figure_size=(14, 10))
        + labs(title='F1 Score of Acceptable Prediction vs. Human Annotation by Criterion',
               x='Model',
               y='F1 Score',
               fill='Type')
    )
    output_path = 'figures/f1_score_faceted.png'
    ggsave(chart, filename=output_path, dpi=300)
    print(f"F1 chart saved to {output_path}")

def generate_precision_recall_chart(df):
    """Generate the faceted precision and recall comparison bar chart."""
    df_melted = df.melt(id_vars=['model', 'type', 'criterion'], 
                        value_vars=['precision', 'recall'], 
                        var_name='metric', value_name='score')

    chart = (
        ggplot(df_melted, aes(x='model', y='score', fill='metric'))
        + geom_bar(stat='identity', position='dodge')
        + facet_wrap('~criterion', ncol=2)
        + theme(axis_text_x=element_text(angle=60, hjust=1, size=8), figure_size=(14, 10))
        + labs(title='Precision and Recall of Acceptable Prediction vs. Human Annotation by Criterion',
               x='Model',
               y='Score',
               fill='Metric')
    )
    output_path = 'figures/precision_recall_faceted.png'
    ggsave(chart, filename=output_path, dpi=300)
    print(f"Precision-Recall chart saved to {output_path}")

def main():
    """Calculate all metrics and generate all charts."""
    all_results = []
    file_paths = glob.glob('spanish_rosie_evals/*.json')

    for file_path in file_paths:
        model_name = shorten_model_name(os.path.basename(file_path))
        file_type = 'Single Model'
        if 'ensemble' in os.path.basename(file_path):
            file_type = 'Ensemble'
        elif 'dafe' in os.path.basename(file_path):
            file_type = 'DAFE'

        metrics = calculate_metrics(file_path)
        for metric_data in metrics:
            all_results.append({
                'model': model_name, 
                'f1_score': metric_data['f1_score'], 
                'precision': metric_data['precision'], 
                'recall': metric_data['recall'], 
                'type': file_type, 
                'criterion': metric_data['criterion']
            })

    df = pd.DataFrame(all_results)

    # Sort models by average F1 score across all criteria for consistent ordering
    avg_scores = df.groupby('model')['f1_score'].mean().sort_values(ascending=True).index
    df['model'] = pd.Categorical(df['model'], categories=avg_scores, ordered=True)

    os.makedirs('figures', exist_ok=True)
    generate_f1_chart(df)
    generate_precision_recall_chart(df)

if __name__ == '__main__':
    main()
