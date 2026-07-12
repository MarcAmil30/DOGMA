from proto_language.core import Segment, Construct
from proto_language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig

# Create a segment and construct
segment = Segment(length=50, sequence_type="dna")
construct = Construct(segments=[segment])

# Set up a generator
generator = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig())
generator.assign(segment)

print(f"Segment: {segment.sequence_length}bp {segment.sequence_type}")
print("Installation successful!")
