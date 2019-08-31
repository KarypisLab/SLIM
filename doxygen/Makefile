DOXYGEN_CMD=doxygen

all: documentation.dox Doxyfile
	$(DOXYGEN_CMD) Doxyfile
	cd latex && make

clean: 
	rm -rf html latex


