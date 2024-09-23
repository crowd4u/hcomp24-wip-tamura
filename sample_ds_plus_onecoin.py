from crowdkit.datasets import load_dataset

from ds_plus_onecoin_model import DSPlusOneCoin

df, gt = load_dataset('relevance-2')
model = DSPlusOneCoin(smooth=1000) # The smooth parameter is $S$ in the paper.
result = model.fit_predict(df)
