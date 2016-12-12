#!/bin/bash
cd "$(dirname "$0")"
fname=$(ls|grep rockspec|grep -v scm | sort -r -V|head -n1)
echo "Last known version: $fname" 
luarocks new_version $fname

new_fname=$(ls|grep rockspec|grep -v scm | sort -r -V|head -n1)
new_version=$(echo $new_fname | cut -f2,3,4,5 -d'-'|sed -e 's/.rockspec//g')
echo "new rockspec: $new_fname"
echo "new version: $new_version"
git add $new_fname
git commit -m "Cutting version $new_version"
git branch $new_version

git push origin master:master
git push origin $new_version:$new_version

git clone https://github.com/torch/rocks
cp $new_fname rocks/
cd rocks
th make-manifest.lua
git add $new_fname
git commit -am "adding rockspec $new_fname"
git push
cd ..
rm -rf rocks
cd ..

