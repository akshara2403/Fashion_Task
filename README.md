Steps to run the application
Clone the github repository 
   git clone https://github.com/akshara2403/Fashion_Task.git
Build a docker image
   docker build -t your-image-name
Run the Container using the following command
   docker run -p $port:80 image_name
Now run the container on your localhost
   http://localhost:$port/
Use Swagger docs by typing /docs , upload image and click on execute to get the output
