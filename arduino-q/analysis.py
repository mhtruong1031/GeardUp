# Main Analysis Pipeline

import pipelines.preprocessing as preprocessing

class AnalysisPipeline:
    def __init__(self):
        self.preprocessing_pipeline = preprocessing.PreprocessingPipeline()

    def analyze(self, data):
        data = self.preprocessing_pipeline.preprocess(data)
        pass