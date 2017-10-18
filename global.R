##Packages
library(shiny)
library(readxl)
library(leaflet)
library(scales)
library(lattice)
library(plyr)
library(dplyr)
library(ggvis)
library(rbokeh)
library(plotly)
library(shinydashboard)
library(highcharter)
library(d3Tree)
library(reshape2)
library(stringr)
library(DT)
library(RColorBrewer)
##Dataset
Data <<- read_excel("~/PeTaL Project/Going into Dash/measured data.xlsx") 
M<-Data%>%count_(c("Status","Class","Order","Family","Species"))
Q<-read_excel("~/PeTaL Project/Testing/RearrangedAvgData.xlsx")
m<-Q%>%data.frame%>%mutate(NEWCOL=NA)%>%distinct

##Categories
Order<<-Data$Order
Family<<-Data$Family
Species<<-Data$Species
Environment<<-Data$Environment
DeepTime<<-Data$`Geological time`
Collectiom<<-Data$Collection

##Data Fixes
Data$`Link to functions` <- paste0("<a href='",Data$`Link to functions`,"'target='_blank'>","Learn more about this Specimen","</a>")
Data$Sex[Data$Sex==2]<-"Female"
Data$Sex[Data$Sex==1]<-"Male"
Data$Sex[Data$Sex==0]<-"Unidentified"
Data[,33:93]<-round(Data[,33:93],digits = 3)
Data$Date<-as.Date(Data$Date)
M$n<-as.numeric(M$n)
M$Family[M$Family=="Unidentified"]<-"Unidentified1"
M$Species[M$Species=="Unidentified"]<-"Unidentified2"

##Data Classifaction   
Flight.Animals<<-Data[Data$Classifaction=="Flight animal",]
Land.Animals<<-Data[Data$Classifaction=="Land animal",]
Water.Animals<<-Data[Data$Classifaction=="Marine animal",]
Plants<<-Data[Data$Classifaction=="Plant",]
Micro<<-Data[Data$Classifaction=="Micro Organism",]
Extremophiles<<-Data[Data$Classifaction=="Extremophile",]
Non<<-Data[Data$Classifaction=="Non-Living",]

##Flight.Animals Orders   
Anisoptera<<-Flight.Animals[Flight.Animals$Order=="Anisoptera",]
Zygoptera<<-Flight.Animals[Flight.Animals$Order=="Zygoptera",]
Hymenoptera<<-Flight.Animals[Flight.Animals$Order=="Hymenoptera",]
Pterosauria<<-Flight.Animals[Flight.Animals$Order=="Pterosauria",]
Diptera<<-Flight.Animals[Flight.Animals$Order=="Diptera",]
Hemiptera<<-Flight.Animals[Flight.Animals$Order=="Hemiptera",]
Saurischia<<-Flight.Animals[Flight.Animals$Order=="Saurischia",]
Confuciusornithiformes<<-Flight.Animals[Flight.Animals$Order=="Confuciusornithiformes",]
Odonata<<-merge(merge(Anisoptera,Flight.Animals[Flight.Animals$Order=="Odonata",],all = T),Zygoptera, all=T)

##Water.Animals Orders
Decapoda<<-Water.Animals[Water.Animals$Order=="Decapoda",]
Xiphosurida<<-Water.Animals[Water.Animals$Order=="Xiphosurida",]
Eurypterida<<-Water.Animals[Water.Animals$Order=="Eurypterida",]
Eumalacostraca<<-Water.Animals[Water.Animals$Order=="Eumalacostraca",]
Ptychopariida<<-Water.Animals[Water.Animals$Order=="Ptychopariida",]
Redlichiida<<-Water.Animals[Water.Animals$Order=="Redlichiida",]
Corynexochida<<-Water.Animals[Water.Animals$Order=="Corynexochida",]
Agnostida<<-Water.Animals[Water.Animals$Order=="Agnostida",]
Asaphida<<-Water.Animals[Water.Animals$Order=="Asaphida",]
Phacopida<<-Water.Animals[Water.Animals$Order=="Phacopida",]
Proetida<<-Water.Animals[Water.Animals$Order=="Proetida",]

##Class
Insecta<<-Data[Data$Class=="Insecta",]
Reptilia<<-Data[Data$Class=="Reptilia",]
Malacostraca<<-Data[Data$Class=="Malacostraca",]
Trilobita<<-Data[Data$Class=="Trilobita",]
Merostomata<<-Data[Data$Class=="Merostomata",]
Xiphosura<<-Data[Data$Class=="Xiphosura",]  

##Phylum
Arthropoda<<-merge(merge(merge(merge(Insecta,Malacostraca,all=T),Trilobita,all=T),Merostomata,all=T),Xiphosura,all=T)
