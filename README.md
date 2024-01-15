# Steps to run the application
## Clone the github repository 
'''bash
git clone https://github.com/akshara2403/Fashion_Task.git
'''
## Build a docker image
'''bash  
docker build -t your-image-name
'''
## Run the Container using the following command
'''bash
docker run -p $port:80 image_name
'''
## Now run the container on your localhost
'''bash
http://localhost:$port/
'''
# Use Swagger docs by typing /docs , upload image and click on execute to get the output
