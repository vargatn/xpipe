############################################################
#
# MAKE the proclens PIPELINE or API
#
############################################################

# xshear settings
XPATH = submodules/xshear
XBRANCH = redmapper


############################################################
# install the python module and dependencies

pipeline: xshear logpath
	python setup.py develop --user

# installing the latest xshear
xshear:
	git submodule init $(XPATH)
	git submodule update $(XPATH)
	(cd $(XPATH) && git checkout $(XBRANCH) && git pull)
	make -C $(XPATH)  install prefix=./

# writes absolute path of program direcory
logpath:
	echo project_path: $(CURDIR)/ > $(HOME)/.proclens.yml


# Clears submodules
deinit:
	git submodule deinit -f .