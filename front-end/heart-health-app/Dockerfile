# Use an official Node.js runtime as a base image
FROM node:latest

# Set the working directory in the container
WORKDIR /user/src/app

# Copy package.json and package-lock.json to the working directory
COPY front-end/heart-health-app/package*.json ./

#COPY package*.json ./

# Install project dependencies
RUN npm install

# Copy the entire project to the working directory
COPY . .

# Expose port 3000 (default for React development server)
EXPOSE 3000


# Start the development server
CMD ["npm", "start"]
