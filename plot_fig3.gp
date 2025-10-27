set terminal pngcairo

if (ARGC < 1) {
  print "Usage: gnuplot -c plot_gpu.gp <csv_file>"
  exit
}

filename = system(sprintf("basename -s .csv %s", ARGV[1]))
file = substr(filename, 5,strlen(filename))
set output filename.'_fig.png'
set datafile separator ','
set style data histograms
set style histogram cluster
set style fill solid border -1
set boxwidth 0.9
set key autotitle columnhead
set xtics rotate by -45
set logscale y 
set ylabel "Edges per second" 
set title "fig3 with " . file
set yrange [1000000:*]
plot ARGV[1] using 2:xtic(1) with histogram title columnheader(2)
print "Finished ", file
