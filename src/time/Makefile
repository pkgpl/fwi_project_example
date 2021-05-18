NP=4
HOST=localhost:4

DEVICE=cuda

DATADIR=$(PROJECT_DIR)/data/data_marm
FRDDIR=$(PROJECT_DIR)/data/forward

# fdm order
ORDER=2

VMIN=1.5
VMAX=5.5

COMMENT=Adam

LR=0.02
OPTIM=adam
MAX_EPOCHS=500

inv_cuda:
	horovodrun -np $(NP) -H $(HOST) python $(PROJECT_DIR)/bin/inv_torch.py \
		--name "Conventional FWI" \
		--comment $(COMMENT) \
		--device cuda \
		--fvel $(DATADIR)/vinit.rsf \
		--fwav $(DATADIR)/wavelet.rsf \
		--fshot $(DATADIR)/shot.rsf \
		--ftrue $(FRDDIR)/seismo. \
		--order $(ORDER) \
		--vmin $(VMIN) \
		--vmax $(VMAX) \
		--lr $(LR) \
		--optimizer $(OPTIM) \
		--skip_output 10 \
		--max_epochs $(MAX_EPOCHS)

