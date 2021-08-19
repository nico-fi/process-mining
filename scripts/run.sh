#!/bin/sh

if [ "$1" = "SPL" ]
then
	java -cp "lib/*" au.edu.unimelb.services.ServiceProvider "SMD" 0.1 0.0 "false" $2 $3
else
	export IMPORTLOG=$2
	export EXPORTMODEL=$3.pnml
	java -da -Xmx8G -classpath "lib/*" -Djava.library.path=./lib -Djava.util.Arrays.useLegacyMergeSort=true org.processmining.contexts.cli.CLI -f Scripts/ILP.txt
fi
