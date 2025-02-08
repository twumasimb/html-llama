from src.pipeline.dataset_gen import DatasetGenerator

# Generate dataset
generator = DatasetGenerator()
generator.generate(size=1000)
generator.save()