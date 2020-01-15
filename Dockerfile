# Load dependencies from petal-dep
FROM sciapps.grc.nasa.gov:5000/petal/petal-deps:0.2

# Copy python requirements to install to image
COPY . /petal

# Set the working directory
WORKDIR /petal

# Expose port to host
EXPOSE 5000

# Run petal
CMD ["python", "app.py"]
