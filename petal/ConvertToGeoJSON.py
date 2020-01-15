import csv
# need to update to output with no template and no comma on last iteration, plus skipping data without coordinates

# Read in raw data from csv
rawData = csv.reader(open('data/allData.csv'))


# the template. where data from the csv will be formatted to geojson
template = \
   ''' \
   { "type" : "Feature",
       "geometry" : {
           "type" : "Point",
           "coordinates" : [%s, %s]},
       "properties" : { "Species" : "%s", "Class": "%s", "Order": "%s", "Common_Name" : "%s", "Specimen" : "%s", "Author": "%s", "Year": "%s", "Collection": "%s", "Location": "%s", "color" : "%s"}
       }
   '''


# the head of the geojson file
output = \
   ''' \

{ "type" : "Feature Collection",
   "features" : [
   '''


# loop through the csv by row skipping the first
next(rawData)
iter = 0
for row in rawData:
   #iter += 1
   #if iter >= 2:
   Species = row[8]
   Specimen = row[0]
   Author = row[9]
   Year = row[10]
   lat = row[13]
   lon = row[14]
   Collection = row[1]
   Common_Name = row[6]
   Order = row[5]
   Class = row[4]
   Location = row[12]
   color = row[15]
   output += template % (lon, lat, Species, Class, Order, Common_Name, Specimen, Author, Year, Collection, Location, color)+ ","
if iter - 1:
   output += template % (lon, lat, Species, Class, Order, Common_Name, Specimen, Author, Year, Collection, Location, color)
# the tail of the geojson file
output += \
   ''' \
   ]

}
   '''


# opens an geoJSON file to write the output
outFileHandle = open("data/Datatest2.geojson", "w")
outFileHandle.write(output)
outFileHandle.close()