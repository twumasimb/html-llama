from src.pipeline.dataset_gen import DatasetGenerator

# Generate dataset
generator = DatasetGenerator(output_file="dataset_10k.json") #dataset_2_5k
generator.generate(size=10000)
generator.save()