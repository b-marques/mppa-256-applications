MPPADIR=/usr/local/k1tools

$MPPADIR/bin/k1-power -gpdf -o tmp/ --period=20 -- $MPPADIR/bin/k1-jtag-runner --multibinary=output/bin/convolution-async.img --exec-multibin=IODDR0:convolution-async-master -- $1 $2 $3 $4 $5 $6 $7 $8
mkdir -p tests/Experiments/convolutionExperimentsEUROPAR/energy$1$3$7/
mv tmp/* tests/Experiments/convolutionExperimentsEUROPAR/energy$1$3$7/
rm -r tmp/
