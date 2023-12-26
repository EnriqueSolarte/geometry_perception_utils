img_dir=$1
img_output=$2
cd ${img_dir}
echo "Creating gif from ${img_dir} to ${img_output}"
convert -delay 30 $(ls ${img_dir} | sort -V) ${img_output}
