reset

name  = ARG1

out     = 'plots/ch_'.name.'_dist.eps'
#mytitle = "Patagon (only filter)\nCircunference"
mytitle = "Filters using a ".name." distribution"

set autoscale # scale axes automatically
set term postscript eps color blacktext "Courier" 24
set output out
#set title mytitle font "Courier, 22"

#set ylabel 'Log Time [ms]' rotate by 90 offset 1
set ylabel 'Speedup' rotate by 90 offset 1
#set log y


set xlabel '# Points'
set font "Courier, 30"
set pointsize 1.0
#set xtics format "%s"
set yrange [0:50]
#set yrange [0:2]

set style line 1 lt 1 lc rgb 'forest-green' dt 1    pt 5    pi -6   lw 2 # green   
set style line 2 lt 2 lc rgb 'black'        dt 2    pt 2    pi -6   lw 2 # orange
set style line 3 lt 3 lc rgb 'web-blue'     dt 6    pt 6    pi -6   lw 2 # blue
set style line 4 lt 4 lc rgb 'red'          dt 5    pt 11   pi -6   lw 2 # purple
set style line 5 lt 1 lc rgb '#77ac30'              pt 13   pi -6   lw 2 # green
set style line 6 lt 1 lc rgb '#4dbeee'              pt 4    pi -6   lw 2 # light-blue
set style line 7 lt 1 lc rgb '#a2142f'              pt 8    pi -6   lw 2 # red
set style line 8 lt 8 lc rgb 'pink'                 pt 7    pi -6   lw 2 # pink

set key Left top left reverse samplen 3.0 font "Courier,22" spacing 1 
#set key Left bot right reverse samplen 3.0 font "Courier,18" spacing 1 


data = 'data/ch_'.name.'_dist.dat'

plot    data              using 1:($2/$7)        title "[CPU] CGAL:ch-graham-andrew"    with lp ls 6, \
        data              using 1:($2/$12)       title "[CPU] cpu-manhattan"            with lp ls 1, \
        data              using 1:($2/$22)       title "[GPU] proposed-filter"                 with lp ls 2, \
        data              using 1:($2/$27)       title "[GPU] cub-flagged"              with lp ls 3, \
        data              using 1:($2/$32)       title "[GPU] thrust-scan"              with lp ls 4, \
        data              using 1:($2/$37)       title "[GPU] thrust-copy"              with lp ls 5