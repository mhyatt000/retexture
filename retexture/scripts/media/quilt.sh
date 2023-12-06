
# 2x2 quilt
ffmpeg -i $1 -i $2 -i $3 -i $4 -filter_complex "[0:v][1:v]hstack=inputs=2[top]; [2:v][3:v]hstack=inputs=2[bottom]; [top][bottom]vstack=inputs=2" output.gif

# 1x4 quilt
# ffmpeg -i $1 -i $2 -i $3 -i $4 -filter_complex "[0:v][1:v][2:v][3:v]hstack=inputs=4[v]" -map "[v]" output.gif
