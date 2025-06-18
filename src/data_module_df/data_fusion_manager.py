# data_fusion_manager.py
import numpy as np
import pandas as pd

class FusionManager:
    @staticmethod
    def fuse_predictions(text_probs, image_probs, strategy='mean'):
        if strategy == 'mean':
            return (text_probs + image_probs) / 2
        elif strategy == 'product':
            fused = text_probs * image_probs
            return fused / fused.sum(axis=1, keepdims=True)
        elif strategy == 'weighted':
            return text_probs * 0.6 + image_probs * 0.4
        elif strategy == 'confidence_weighted':
            text_conf = np.max(text_probs, axis=1, keepdims=True)
            image_conf = np.max(image_probs, axis=1, keepdims=True)
            total = text_conf + image_conf
            return text_probs * (text_conf / total) + image_probs * (image_conf / total)
        else:
            raise ValueError(f"Fusion strategy '{strategy}' not supported")

    @staticmethod
    def create_combined_report(y_true, text_preds, image_preds, fused_preds, text_probs, image_probs, fused_probs):
        report = []
        for i in range(len(y_true)):
            report.append({
                'true': y_true[i],
                'text_pred': text_preds[i],
                'image_pred': image_preds[i],
                'fused_pred': fused_preds[i],
                'text_conf': np.max(text_probs[i]),
                'image_conf': np.max(image_probs[i]),
                'fused_conf': np.max(fused_probs[i]),
                'agree_text_image': text_preds[i] == image_preds[i],
                'correct_fusion': fused_preds[i] == y_true[i]
            })
        return pd.DataFrame(report)
