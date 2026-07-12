import cptac

# See available cancer datasets
print(cptac.list_datasets())

# Example: load a cancer cohort
cohort = cptac.Ccrcc()   # kidney cancer example

clinical = cohort.get_clinical()
mutations = cohort.get_somatic_mutation()
transcriptomics = cohort.get_transcriptomics()
proteomics = cohort.get_proteomics()
phosphoproteomics = cohort.get_phosphoproteomics()

print(clinical.head())
print(mutations.head())
print(transcriptomics.shape)
print(proteomics.shape)
