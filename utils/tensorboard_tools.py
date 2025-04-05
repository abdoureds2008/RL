def setup_tensorboard(log_dir='logs'):
    writer = create_file_writer(log_dir)
    return writer

def log_metrics(writer, metrics, step):
    with writer.as_default():
        for k,v in metrics.items():
            tf.summary.scalar(k, v, step=step)
        writer.flush()

def shap_explain(agent, sample):
    if not hasattr(agent, 'model'):
        print("No SHAP for continuous or ensemble agent.")
        return
    explainer = shap.DeepExplainer(agent.model, sample)
    shap_vals = explainer.shap_values(sample)
    shap.summary_plot(shap_vals, sample)
    return shap_vals