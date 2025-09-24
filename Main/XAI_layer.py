class WaferXAIAnalyzer:
    """FIXED: XAI analyzer with better error handling"""

    def __init__(self, model):
        self.model = model
        self.gradcam = None
        if XAI_AVAILABLE:
            try:
                self.gradcam = Gradcam(model,
                                     model_modifier=ReplaceToLinear(),
                                     clone=True)
            except Exception as e:
                print(f"Grad-CAM initialization failed: {e}")
                self.gradcam = None

    def generate_gradcam_heatmap(self, images, class_indices, layer_name=None):
        """Generate Grad-CAM heatmaps with improved error handling"""
        if not XAI_AVAILABLE or self.gradcam is None:
            return None

        try:
            if isinstance(class_indices, np.ndarray):
                class_indices = class_indices.tolist()
            elif not isinstance(class_indices, list):
                class_indices = [class_indices]

            score = CategoricalScore(class_indices)
            heatmaps = self.gradcam(score, images, penultimate_layer=-1)
            return heatmaps
        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")
            return None

    def analyze_model_focus(self, client_data, num_samples=5):
        """FIXED: Analyze model focus with better data handling"""
        if not XAI_AVAILABLE or self.gradcam is None:
            return {"focus_score": 1.0, "defect_coverage": 0.5}

        focus_scores = []
        sample_count = 0

        try:
            for batch in client_data.take(2):
                if sample_count >= num_samples:
                    break

                images, labels = batch
                batch_size = min(images.shape[0], num_samples - sample_count)

                if batch_size <= 0:
                    continue

                # Take subset
                images_subset = images[:batch_size]

                # Get predictions
                predictions = self.model.predict(images_subset, verbose=0)
                predicted_classes = np.argmax(predictions, axis=1).tolist()

                # Generate heatmaps
                heatmaps = self.generate_gradcam_heatmap(images_subset, predicted_classes)

                if heatmaps is not None:
                    for i, heatmap in enumerate(heatmaps):
                        if sample_count >= num_samples:
                            break

                        # Calculate focus metrics
                        max_activation = np.max(heatmap)
                        mean_activation = np.mean(heatmap)
                        focus_ratio = max_activation / (mean_activation + 1e-8)

                        # Simple defect coverage metric
                        h, w = heatmap.shape
                        center_region = heatmap[h//4:3*h//4, w//4:3*w//4]
                        center_activation = np.mean(center_region)
                        defect_coverage = min(center_activation / (mean_activation + 1e-8), 2.0)

                        focus_scores.append({
                            'focus_ratio': focus_ratio,
                            'defect_coverage': defect_coverage
                        })

                        sample_count += 1

        except Exception as e:
            print(f"XAI analysis error: {e}")
            return {"focus_score": 1.0, "defect_coverage": 0.5}

        if focus_scores:
            avg_focus = np.mean([s['focus_ratio'] for s in focus_scores])
            avg_coverage = np.mean([s['defect_coverage'] for s in focus_scores])
            return {"focus_score": avg_focus, "defect_coverage": avg_coverage}

        return {"focus_score": 1.0, "defect_coverage": 0.5}
