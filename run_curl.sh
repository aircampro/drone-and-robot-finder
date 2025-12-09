# runs the curl on the image processing cloud with the picture file specified as argument 1
if (($# == 1)) ;then
   base64 $1 | curl -d @- "https://detect.roboflow.com/drone2-epzxe/7?api_key=Mqvor12zsKc38eyJUP97"
else
   echo "no filename passed to analyse"
fi
