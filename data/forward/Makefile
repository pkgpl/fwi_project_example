NP=4
HOST=localhost:4

DATADIR=../data_marm
FRDDIR=../forward

# fdm order
ORDER=8


forward_cuda:
	horovodrun -np $(NP) -H $(HOST) python $(PROJECT_DIR)/bin/forward_modeling.py \
		--device cuda \
		--fvel $(DATADIR)/marm.rsf \
		--fwav $(DATADIR)/wavelet.rsf \
		--fshot $(DATADIR)/shot.rsf \
		--order $(ORDER) \
		--outdir $(FRDDIR)

