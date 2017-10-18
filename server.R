library(shiny)
library(shinydashboard)
library(DT)
library(cluster)

shinyServer(function(input, output, session) {
  output$frameS <- renderUI({
    tags$iframe(src="Citizens_of_science_Carousel.html", height=465, width="100%")
  })
  output$Net <- renderUI({
    tags$iframe(src="Network.html", height=500, width="100%",style="border:0",padding(0,0,0,0))
  })
  observe({
    AFamily <- if (is.null(input$AOrder)) character(0) else {
      filter(Data, Order %in% input$AOrder) %>%
        `$`('Family') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$AFamily[input$AFamily %in% Family])
    updateSelectInput(session, "AFamily", choices = AFamily,
                      selected = stillSelected)
  })
  observe({
    ASpecies <- if (is.null(input$AOrder)) character(0) else {
      Data %>%
        filter(Order %in% input$AOrder,
               is.null(input$AFamily) | Family %in% input$AFamily) %>%
        `$`('Species') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$ASpecies[input$ASpecies %in% Species])
    updateSelectInput(session, "ASpecies", choices = ASpecies,
                      selected = stillSelected)
  })
  filtered_tbl_1_data <- reactive({
    Data %>% 
      filter(
        is.null(input$Collection) | Collection %in% input$Collection,
        is.null(input$DeepTime) | DeepTime %in% input$DeepTime,
        is.null(input$AOrder) | Order %in% input$AOrder,
        is.null(input$AFamily) | Family %in% input$AFamily,
        is.null(input$ASpecies) | Species %in% input$ASpecies
      )
  })
  output$progressBox <- renderInfoBox({
    infoBox(
      "Specimens in search", value=nrow(filtered_tbl_1_data()), icon = icon("list"),
      color = if (nrow(filtered_tbl_1_data()) >= 50) "red" else "green"
    )
  })
  output$tbl_1 <- DT::renderDataTable({
    DT::datatable(filtered_tbl_1_data(), escape = FALSE,
                  extensions = c('FixedColumns','Scroller','KeyTable','ColReorder','Buttons'), 
                  options = list(bInfo=F,
                                 dom = 'Bfrtip',
                                 scrollX = TRUE,
                                 #fixedColumns = list(leftColumns = 2),
                                 pageLength = 80,
                                 lengthMenu = c(10, 15, 20), 
                                 deferRender = TRUE,
                                 scrollY = 360,
                                 scroller = TRUE,
                                 keys = TRUE,
                                 buttons = list('copy', 'csv', 'excel', 'pdf', 'print'),
                                 colReorder = TRUE))
  }, server = FALSE)
  observeEvent(input$tbl_1_rows_selected, {
    rows <- input$tbl_1_rows_selected
    output$text <- renderText({paste("Functions of ", unlist(filtered_tbl_1_data()[rows,9]), "\n")})
    updateTabItems(session, "tabs", selected = "dataReceiveTest")
  })
  
  eventReactive(input$go, {
    #output$pam.plot<- plot(pam(Data[40,13:14], 7))
    cor.plot<- plot(lm(Status ~ `Geological time` + Order, data= Data[222,]))
    updateTabItems(session, "tabs", selected = "AnalysisTK")
  })
  ##Pop ups and Icons add for each Order and Catagory
  output$map <- renderLeaflet({
    
    Dot_popup_Data <- {paste0("<strong>Specimen: </strong>", 
                              Data$Specimen, 
                              "<br><strong>Common Name: </strong>", 
                              Data$`Common Name`, 
                              "<br><strong>Family: </strong>", 
                              Data$Family, 
                              "<br><strong>Habitat: </strong>", 
                              Data$Habitat,
                              "<br><strong>Notes: </strong>",
                              Data$Notes,
                              "<br>",
                              Data$`Link to functions`)}
    
    Dot_popup_Flight <- {paste0("<strong>Specimen: </strong>", 
                                Flight.Animals$Specimen, 
                                "<br><strong>Common Name: </strong>", 
                                Flight.Animals$`Common Name`, 
                                "<br><strong>Family: </strong>", 
                                Flight.Animals$Family, 
                                "<br><strong>Habitat: </strong>", 
                                Flight.Animals$Habitat,
                                "<br><strong>Notes: </strong>",
                                Flight.Animals$Notes,
                                "<br>",
                                Flight.Animals$`Link to functions`)}
    Dot_popup_Ani <- {paste0("<strong>Specimen: </strong>", 
                             Anisoptera$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Anisoptera$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Anisoptera$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Anisoptera$Habitat,
                             "<br><strong>Notes: </strong>",
                             Anisoptera$Notes,
                             "<br>",
                             Anisoptera$`Link to functions`)}
    Dot_popup_Zyg <- {paste0("<strong>Specimen: </strong>", 
                             Zygoptera$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Zygoptera$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Zygoptera$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Zygoptera$Habitat,
                             "<br><strong>Notes: </strong>",
                             Zygoptera$Notes,
                             "<br>",
                             Zygoptera$`Link to functions`)}
    Dot_popup_Hym <- {paste0("<strong>Specimen: </strong>", 
                             Hymenoptera$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Hymenoptera$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Hymenoptera$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Hymenoptera$Habitat,
                             "<br><strong>Notes: </strong>",
                             Hymenoptera$Notes,
                             "<br>",
                             Hymenoptera$`Link to functions`)}
    Dot_popup_Pte <- {paste0("<strong>Specimen: </strong>", 
                             Pterosauria$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Pterosauria$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Pterosauria$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Pterosauria$Habitat,
                             "<br><strong>Notes: </strong>",
                             Pterosauria$Notes,
                             "<br>",
                             Pterosauria$`Link to functions`)}
    Dot_popup_Oda <- {paste0("<strong>Specimen: </strong>", 
                             Odonata$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Odonata$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Odonata$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Odonata$Habitat,
                             "<br><strong>Notes: </strong>",
                             Odonata$Notes,
                             "<br>",
                             Odonata$`Link to functions`)}
    Dot_popup_Dip <- {paste0("<strong>Specimen: </strong>", 
                             Diptera$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Diptera$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Diptera$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Diptera$Habitat,
                             "<br><strong>Notes: </strong>",
                             Diptera$Notes,
                             "<br>",
                             Diptera$`Link to functions`)}
    Dot_popup_Hem <- {paste0("<strong>Specimen: </strong>", 
                             Hemiptera$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Hemiptera$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Hemiptera$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Hemiptera$Habitat,
                             "<br><strong>Notes: </strong>",
                             Hemiptera$Notes,
                             "<br>",
                             Hemiptera$`Link to functions`)}
    Dot_popup_Sau <- {paste0("<strong>Specimen: </strong>", 
                             Saurischia$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Saurischia$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Saurischia$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Saurischia$Habitat,
                             "<br><strong>Notes: </strong>",
                             Saurischia$Notes,
                             "<br>",
                             Saurischia$`Link to functions`)}
    Dot_popup_Con <- {paste0("<strong>Specimen: </strong>", 
                             Confuciusornithiformes$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Confuciusornithiformes$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Confuciusornithiformes$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Confuciusornithiformes$Habitat,
                             "<br><strong>Notes: </strong>",
                             Confuciusornithiformes$Notes,
                             "<br>",
                             Confuciusornithiformes$`Link to functions`)}
    
    Dot_popup_Land <- {paste0("<strong>Specimen: </strong>", 
                              Land.Animals$Specimen, 
                              "<br><strong>Common Name: </strong>", 
                              Land.Animals$`Common Name`, 
                              "<br><strong>Family: </strong>", 
                              Land.Animals$Family, 
                              "<br><strong>Habitat: </strong>", 
                              Land.Animals$Habitat,
                              "<br><strong>Notes: </strong>",
                              Land.Animals$Notes,
                              "<br>",
                              Land.Animals$`Link to functions`)}
    
    Dot_popup_Marine <- {paste0("<strong>Specimen: </strong>", 
                                Water.Animals$Specimen, 
                                "<br><strong>Common Name: </strong>", 
                                Water.Animals$`Common Name`, 
                                "<br><strong>Family: </strong>", 
                                Water.Animals$Family, 
                                "<br><strong>Habitat: </strong>", 
                                Water.Animals$Habitat,
                                "<br><strong>Notes: </strong>",
                                Water.Animals$Notes,
                                "<br>",
                                Water.Animals$`Link to functions`)}
    Dot_popup_Dec <- {paste0("<strong>Specimen: </strong>", 
                             Decapoda$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Decapoda$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Decapoda$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Decapoda$Habitat,
                             "<br><strong>Notes: </strong>",
                             Decapoda$Notes,
                             "<br>",
                             Decapoda$`Link to functions`)}
    Dot_popup_Xip <- {paste0("<strong>Specimen: </strong>", 
                             Xiphosurida$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Xiphosurida$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Xiphosurida$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Xiphosurida$Habitat,
                             "<br><strong>Notes: </strong>",
                             Xiphosurida$Notes,
                             "<br>",
                             Xiphosurida$`Link to functions`)}
    Dot_popup_Eur <- {paste0("<strong>Specimen: </strong>", 
                             Eurypterida$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Eurypterida$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Eurypterida$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Eurypterida$Habitat,
                             "<br><strong>Notes: </strong>",
                             Eurypterida$Notes,
                             "<br>",
                             Eurypterida$`Link to functions`)}
    Dot_popup_Eum <- {paste0("<strong>Specimen: </strong>", 
                             Eumalacostraca$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Eumalacostraca$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Eumalacostraca$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Eumalacostraca$Habitat,
                             "<br><strong>Notes: </strong>",
                             Eumalacostraca$Notes,
                             "<br>",
                             Eumalacostraca$`Link to functions`)}
    Dot_popup_Pty <- {paste0("<strong>Specimen: </strong>", 
                             Ptychopariida$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Ptychopariida$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Ptychopariida$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Ptychopariida$Habitat,
                             "<br><strong>Notes: </strong>",
                             Ptychopariida$Notes,
                             "<br>",
                             Ptychopariida$`Link to functions`)}
    Dot_popup_Red <- {paste0("<strong>Specimen: </strong>", 
                             Redlichiida$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Redlichiida$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Redlichiida$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Redlichiida$Habitat,
                             "<br><strong>Notes: </strong>",
                             Redlichiida$Notes,
                             "<br>",
                             Redlichiida$`Link to functions`)}
    Dot_popup_Cor <- {paste0("<strong>Specimen: </strong>", 
                             Corynexochida$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Corynexochida$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Corynexochida$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Corynexochida$Habitat,
                             "<br><strong>Notes: </strong>",
                             Corynexochida$Notes,
                             "<br>",
                             Corynexochida$`Link to functions`)}
    Dot_popup_Agn <- {paste0("<strong>Specimen: </strong>", 
                             Agnostida$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Agnostida$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Agnostida$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Agnostida$Habitat,
                             "<br><strong>Notes: </strong>",
                             Agnostida$Notes,
                             "<br>",
                             Agnostida$`Link to functions`)}
    Dot_popup_Asa <- {paste0("<strong>Specimen: </strong>", 
                             Asaphida$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Asaphida$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Asaphida$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Asaphida$Habitat,
                             "<br><strong>Notes: </strong>",
                             Asaphida$Notes,
                             "<br>",
                             Asaphida$`Link to functions`)}
    Dot_popup_Pha <- {paste0("<strong>Specimen: </strong>", 
                             Phacopida$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Phacopida$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Phacopida$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Phacopida$Habitat,
                             "<br><strong>Notes: </strong>",
                             Phacopida$Notes,
                             "<br>",
                             Phacopida$`Link to functions`)}
    Dot_popup_Pro <- {paste0("<strong>Specimen: </strong>", 
                             Proetida$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Proetida$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Proetida$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Proetida$Habitat,
                             "<br><strong>Notes: </strong>",
                             Proetida$Notes,
                             "<br>",
                             Proetida$`Link to functions`)}
    
    Dot_popup_Plants <- {paste0("<strong>Specimen: </strong>", 
                                Plants$Specimen, 
                                "<br><strong>Common Name: </strong>", 
                                Plants$`Common Name`, 
                                "<br><strong>Family: </strong>", 
                                Plants$Family, 
                                "<br><strong>Habitat: </strong>", 
                                Plants$Habitat,
                                "<br><strong>Notes: </strong>",
                                Plants$Notes,
                                "<br>",
                                Plants$`Link to functions`)}
    
    Dot_popup_Micro <- {paste0("<strong>Specimen: </strong>", 
                               Micro$Specimen, 
                               "<br><strong>Common Name: </strong>", 
                               Micro$`Common Name`, 
                               "<br><strong>Family: </strong>", 
                               Micro$Family, 
                               "<br><strong>Habitat: </strong>", 
                               Micro$Habitat,
                               "<br><strong>Notes: </strong>",
                               Micro$Notes,
                               "<br>",
                               Micro$`Link to functions`)}
    
    Dot_popup_Extremophiles <- {paste0("<strong>Specimen: </strong>", 
                                       Extremophiles$Specimen, 
                                       "<br><strong>Common Name: </strong>", 
                                       Extremophiles$`Common Name`, 
                                       "<br><strong>Family: </strong>", 
                                       Extremophiles$Family, 
                                       "<br><strong>Habitat: </strong>", 
                                       Extremophiles$Habitat,
                                       "<br><strong>Notes: </strong>",
                                       Extremophiles$Notes,
                                       "<br>",
                                       Extremophiles$`Link to functions`)}
    
    Dot_popup_Non <- {paste0("<strong>Specimen: </strong>", 
                             Non$Specimen, 
                             "<br><strong>Common Name: </strong>", 
                             Non$`Common Name`, 
                             "<br><strong>Family: </strong>", 
                             Non$Family, 
                             "<br><strong>Habitat: </strong>", 
                             Non$Habitat,
                             "<br><strong>Notes: </strong>",
                             Non$Notes,
                             "<br>",
                             Non$`Link to functions`)}
    
    Dot_popup_Insecta <- {paste0("<strong>Specimen: </strong>", 
                                 Insecta$Specimen, 
                                 "<br><strong>Common Name: </strong>", 
                                 Insecta$`Common Name`, 
                                 "<br><strong>Family: </strong>", 
                                 Insecta$Family, 
                                 "<br><strong>Habitat: </strong>", 
                                 Insecta$Habitat,
                                 "<br><strong>Notes: </strong>",
                                 Insecta$Notes,
                                 "<br>",
                                 Insecta$`Link to functions`)}
    Dot_popup_Reptilia <- {paste0("<strong>Specimen: </strong>", 
                                  Reptilia$Specimen, 
                                  "<br><strong>Common Name: </strong>", 
                                  Reptilia$`Common Name`, 
                                  "<br><strong>Family: </strong>", 
                                  Reptilia$Family, 
                                  "<br><strong>Habitat: </strong>", 
                                  Reptilia$Habitat,
                                  "<br><strong>Notes: </strong>",
                                  Reptilia$Notes,
                                  "<br>",
                                  Reptilia$`Link to functions`)}
    Dot_popup_Malacostraca <- {paste0("<strong>Specimen: </strong>", 
                                      Malacostraca$Specimen, 
                                      "<br><strong>Common Name: </strong>", 
                                      Malacostraca$`Common Name`, 
                                      "<br><strong>Family: </strong>", 
                                      Malacostraca$Family, 
                                      "<br><strong>Habitat: </strong>", 
                                      Malacostraca$Habitat,
                                      "<br><strong>Notes: </strong>",
                                      Malacostraca$Notes,
                                      "<br>",
                                      Malacostraca$`Link to functions`)}
    Dot_popup_Trilobita <- {paste0("<strong>Specimen: </strong>", 
                                   Trilobita$Specimen, 
                                   "<br><strong>Common Name: </strong>", 
                                   Trilobita$`Common Name`, 
                                   "<br><strong>Family: </strong>", 
                                   Trilobita$Family, 
                                   "<br><strong>Habitat: </strong>", 
                                   Trilobita$Habitat,
                                   "<br><strong>Notes: </strong>",
                                   Trilobita$Notes,
                                   "<br>",
                                   Trilobita$`Link to functions`)}
    Dot_popup_Merostomata <- {paste0("<strong>Specimen: </strong>", 
                                     Merostomata$Specimen, 
                                     "<br><strong>Common Name: </strong>", 
                                     Merostomata$`Common Name`, 
                                     "<br><strong>Family: </strong>", 
                                     Merostomata$Family, 
                                     "<br><strong>Habitat: </strong>", 
                                     Merostomata$Habitat,
                                     "<br><strong>Notes: </strong>",
                                     Merostomata$Notes,
                                     "<br>",
                                     Merostomata$`Link to functions`)}
    Dot_popup_Xiphosura <- {paste0("<strong>Specimen: </strong>", 
                                   Xiphosura$Specimen, 
                                   "<br><strong>Common Name: </strong>", 
                                   Xiphosura$`Common Name`, 
                                   "<br><strong>Family: </strong>", 
                                   Xiphosura$Family, 
                                   "<br><strong>Habitat: </strong>", 
                                   Xiphosura$Habitat,
                                   "<br><strong>Notes: </strong>",
                                   Xiphosura$Notes,
                                   "<br>",
                                   Xiphosura$`Link to functions`)}
    
    Dot_Popup_Arthropoda <-{paste0("<strong>Specimen: </strong>", 
                                   Arthropoda$Specimen, 
                                   "<br><strong>Common Name: </strong>", 
                                   Arthropoda$`Common Name`, 
                                   "<br><strong>Family: </strong>", 
                                   Arthropoda$Family, 
                                   "<br><strong>Habitat: </strong>", 
                                   Arthropoda$Habitat,
                                   "<br><strong>Notes: </strong>",
                                   Arthropoda$Notes,
                                   "<br>",
                                   Arthropoda$`Link to functions`)}
    
    Data_icons <- icons(
      iconUrl = Data$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Flight_icons <- icons(
      iconUrl = Flight.Animals$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Land_icons <- icons(
      iconUrl = Land.Animals$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Marine_icons <- icons(
      iconUrl = Water.Animals$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Plants_icons <- icons(
      iconUrl = Plants$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Micro_icons <- icons(
      iconUrl = Micro$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Extremophiles_icons <- icons(
      iconUrl = Extremophiles$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Non_icons <- icons(
      iconUrl = Non$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Dragonfly_icons <- icons(
      iconUrl = Anisoptera$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Damselfly_icons <- icons(
      iconUrl = Zygoptera$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Hymenoptera_icons <- icons(
      iconUrl = Hymenoptera$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Pterosauria_icons <- icons(
      iconUrl = Pterosauria$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Diptera_icons<-icons(
      iconUrl = Diptera$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Odonata_icons<-icons(
      iconUrl = Odonata$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Hemiptera_icons<-icons(
      iconUrl = Hemiptera$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Decapoda_icons <- icons(
      iconUrl = Decapoda$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Xiphosurida_icons <- icons(
      iconUrl = Xiphosurida$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Eurypterida_icons <- icons(
      iconUrl = Eurypterida$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Eumalacostraca_icons <- icons(
      iconUrl = Eumalacostraca$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Ptychopariida_icons <- icons(
      iconUrl = Ptychopariida$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Redlichiida_icons <- icons(
      iconUrl = Redlichiida$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Corynexochida_icons <- icons(
      iconUrl = Corynexochida$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Agnostida_icons <- icons(
      iconUrl = Agnostida$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Asaphida_icons <- icons(
      iconUrl = Asaphida$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Phacopida_icons <- icons(
      iconUrl = Phacopida$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Proetida_icons <- icons(
      iconUrl = Proetida$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Insecta_icons <- icons(
      iconUrl = Insecta$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Reptilia_icons <- icons(
      iconUrl = Reptilia$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Malacostraca_icons <- icons(
      iconUrl = Malacostraca$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Trilobita_icons <-icons(
      iconUrl = Trilobita$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Merostomata_icons <-icons(
      iconUrl = Merostomata$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Xiphosura_icons <-icons(
      iconUrl = Xiphosura$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Saurischia_icons <-icons(
      iconUrl = Saurischia$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Confuciusornithiformes_icons <-icons(
      iconUrl = Confuciusornithiformes$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    Arthropoda_icons <-icons(
      iconUrl = Arthropoda$'Map Icon',
      iconWidth = 40, iconHeight = 40)
    map<- leaflet(options = leafletOptions(zoomControl = T,
                                           minZoom = 2)) %>%
      addMiniMap(toggleDisplay = TRUE,position = c("topleft"))%>%
      addProviderTiles(providers$OpenStreetMap, group= "Default") %>%
      addProviderTiles(providers$Hydda.Full, group= "Rivers") %>% 
      addProviderTiles(providers$Esri.WorldPhysical, group= "Topo") %>% 
      addProviderTiles(providers$Esri.WorldImagery, group= "Imagery") %>% 
      setMaxBounds(lng1=200,lat1=200,lng2=-200,lat2=-200)%>%
      addMarkers(jitter(as.numeric(Organisms()$lon)),jitter(as.numeric(Organisms()$lat)),
                 icon = switch(input$point,
                               "All"=Data_icons,
                               "All Flight Animals"=Flight_icons,
                               "Anisoptera"=Dragonfly_icons,
                               "Zygoptera"=Damselfly_icons,
                               "Hymenoptera"=Hymenoptera_icons,
                               "Pterosauria"=Pterosauria_icons,
                               "Odonata"=Odonata_icons,
                               "Diptera"=Diptera_icons,
                               "Hemiptera"=Hemiptera_icons,
                               "All Land Animals"=Land_icons,
                               "All Water Animals"=Marine_icons,
                               "Decapoda"=Decapoda_icons,
                               "Xiphosurida"=Xiphosurida_icons,
                               "Eurypterida"=Eurypterida_icons,
                               "Eumalacostraca"=Eumalacostraca_icons,
                               "Ptychopariida"=Ptychopariida_icons,
                               "Redlichiida"=Redlichiida_icons,
                               "Corynexochida"=Corynexochida_icons,
                               "Agnostida"=Agnostida_icons,
                               "Asaphida"=Asaphida_icons,
                               "Phacopida"=Phacopida_icons,
                               "Proetida"=Proetida_icons,
                               "All Plants"=Plants_icons,
                               "All Micro"=Micro_icons,
                               "All Extremophiles"=Extremophiles_icons,
                               "All Non-Living"=Non_icons,
                               "Insecta"=Insecta_icons,
                               "Reptilia"=Reptilia_icons,
                               "Malacostraca"=Malacostraca_icons,
                               "Merostomata"=Merostomata_icons,
                               "Trilobita"=Trilobita_icons,
                               "Xiphosura"=Xiphosura_icons,
                               "Saurischia"= Saurischia_icons,
                               "Confuciusornithiformes"=Confuciusornithiformes_icons,
                               "Arthropoda"=Arthropoda_icons),
                 popup=switch(input$point,
                              "All"=Dot_popup_Data,
                              "All Flight Animals"=Dot_popup_Flight,
                              "Anisoptera"=Dot_popup_Ani,
                              "Zygoptera"=Dot_popup_Zyg,
                              "Hymenoptera"=Dot_popup_Hym,
                              "Pterosauria"=Dot_popup_Pte,
                              "Odonata"=Dot_popup_Oda,
                              "Diptera"=Dot_popup_Dip,
                              "Hemiptera"=Dot_popup_Hem,
                              "All Land Animals"=Dot_popup_Land,
                              "All Water Animals"=Dot_popup_Marine,
                              "Decapoda"=Dot_popup_Dec,
                              "Xiphosurida"=Dot_popup_Xip,
                              "Eurypterida"=Dot_popup_Eur,
                              "Eumalacostraca"=Dot_popup_Eum,
                              "Ptychopariida"=Dot_popup_Pty,
                              "Redlichiida"=Dot_popup_Red,
                              "Corynexochida"=Dot_popup_Cor,
                              "Agnostida"=Dot_popup_Agn,
                              "Asaphida"=Dot_popup_Asa,
                              "Phacopida"=Dot_popup_Pha,
                              "Proetida"=Dot_popup_Pro,
                              "All Plants"=Dot_popup_Plants,
                              "All Micro"=Dot_popup_Micro,
                              "All Extremophiles"=Dot_popup_Extremophiles,
                              "All Non-Living"=Dot_popup_Non,
                              "Insecta"=Dot_popup_Insecta,
                              "Reptilia"=Dot_popup_Reptilia,
                              "Malacostraca"=Dot_popup_Malacostraca,
                              "Trilobita"=Dot_popup_Trilobita,
                              "Merostomata"=Dot_popup_Merostomata,
                              "Xiphosura"=Dot_popup_Xiphosura,
                              "Saurischia"=Dot_popup_Sau,
                              "Confuciusornithiformes"=Dot_popup_Con,
                              "Arthropoda"=Dot_Popup_Arthropoda) 
      )%>%
      addLayersControl(baseGroups = c("Default", "Imagery","Rivers"),position = c("topleft"),
                       options = layersControlOptions(collapsed = F,autoZIndex = TRUE))
  })     
  
  Organisms<-reactive({switch(input$point,
                              "All"=Data,
                              "All Flight Animals"=Flight.Animals,
                              "Anisoptera"=Anisoptera,
                              "Zygoptera"=Zygoptera,
                              "Hymenoptera"=Hymenoptera,
                              "Pterosauria"=Pterosauria,
                              "Odonata"=Odonata,
                              "Diptera"=Diptera,
                              "Hemiptera"=Hemiptera,
                              "All Water Animals"=Water.Animals,
                              "Decapoda"=Decapoda,
                              "Xiphosurida"=Xiphosurida,
                              "Eurypterida"=Eurypterida,
                              "Eumalacostraca"=Eumalacostraca,
                              "Ptychopariida"=Ptychopariida,
                              "Redlichiida"=Redlichiida,
                              "Corynexochida"=Corynexochida,
                              "Agnostida"=Agnostida,
                              "Asaphida"=Asaphida,
                              "Phacopida"=Phacopida,
                              "Proetida"=Proetida,
                              "Insecta"=Insecta,
                              "Reptilia"=Reptilia,
                              "Malacostraca"=Malacostraca,
                              "Trilobita"=Trilobita,
                              "Merostomata"=Merostomata,
                              "Xiphosura"=Xiphosura,
                              "Saurischia"=Saurischia,
                              "Confuciusornithiformes"=Confuciusornithiformes,
                              "Arthropoda"=Arthropoda)})
  
  ##Legends
  observe({
    proxy <- leafletProxy("map")
    proxy %>% clearControls()
    if (input$legend) {   
      legend_Wings <- "Color is based on families<br/>
      <img src='Damselfly.png'
      style='width:30px;height:30px;'>Damselfly 
      
      <img src='Dragonfly.png'  
      style='width:30px;height:30px;'>Dragonfly<br/>
      
      <img src='Wasp.png' 
      style='width:30px;height:30px;'>Wasp
      
      <img src='Bee.png' 
      style='width:30px;height:30px;'>Bee<br/> 
      
      <img src='Butterfly.png' 
      style='width:30px;height:30px;'>Butterfly
      
      <img src='Fly.png' 
      style='width:30px;height:30px;'>Fly<br/> 
      
      <img src='Sawfly.png' 
      style='width:30px;height:30px;'>Sawfly
      
      <img src='Pterosaur.png' 
      style='width:30px;height:30px;'>Pterosaur<br/> "
      
      legend_Land <- "Color is based on families<br/>
      <img src='Bear.png'
      style='width:30px;height:30px;'>Bear<br/> "
      
      legend_Plant <- "Color is based on families<br/>
      <img src='Tree.png'
      style='width:30px;height:30px;'>Tree<br/> "
      
      legend_Water <- "Color is based on families<br/>
      <img src='Whale.png'
      style='width:30px;height:30px;'>Whale<br/> 
      
      <img src='Ichthyosaur.png'
      style='width:30px;height:30px;'>Ichthyosaur<br/>"
      
      legend_Extremo <- "Color is based on families<br/>
      <img src='Tardegrade.png'
      style='width:30px;height:30px;'>Tardegrade<br/> "
      
      legend_Micro <- "Color is based on families<br/>
      <img src='Microbe.png'
      style='width:30px;height:30px;'>Microbe<br/> "
      
      legend_Non <- "Coming soon! "
      proxy %>% addControl(html =switch(input$Legend,
                                       "Flight Animals"=legend_Wings,
                                       "Land Animals"=legend_Land,
                                       "Water Animals"=legend_Water,
                                       "Plants"=legend_Plant,
                                       "Extremophiles"=legend_Extremo,
                                       "Micro Life"=legend_Micro,
                                       "Non-Living"=legend_Non),
                          position = "bottomleft")
    }
  })
  ##Graphs  
  ##Plot only points seen
  pointsInBounds2 <- reactive({
    req(input$map_bounds)
    
    bounds <- input$map_bounds
    latRng <- range(bounds$north, bounds$south)
    lngRng <- range(bounds$east, bounds$west)
    
    Organisms<-filter(Organisms(),
                      lat >= latRng[1] & lat <= latRng[2] &
                        lon >= lngRng[1] & lon <= lngRng[2])
    return(Organisms)
  })
  ##Don't plot bogus numbers
  pointsInBounds<-reactive({ 
    DataP<-data.frame(pointsInBounds2())
    colnames(DataP)<-c(colnames(Data))
    DataP[DataP==-1]<-NA
    DataP$`BP (mb) High`[DataP$`BP (mb) High`==1500]<-NA
    DataP$`BP (mb) Low`[DataP$`BP (mb) Low`==1500]<-NA
    DataP$`Wind Low (Km/hr)`[DataP$`Wind Low (Km/hr)`==150]<-NA
    DataP$`Wind High (Km/hr)`[DataP$`Wind High (Km/hr)`==150]<-NA
    DataP$`Temp C Low`[DataP$`Temp C Low`==150]<-NA
    DataP$`Temp C High`[DataP$`Temp C High`==150]<-NA
    DataP$`Humidity High`[DataP$`Humidity High`==150]<-NA
    DataP$`Humidity Low`[DataP$`Humidity Low`==150]<-NA
    DataP$`Percipitation (mm)`[DataP$`Percipitation (mm)`==150]<-NA
    return(DataP)
  })
  
  ##histogram
  output$hist <- renderHighchart({
    plot1<- # If no organisms are in view, don't plot
      if (nrow(pointsInBounds()) == 0)
        return()
    hchart(switch(input$X,"Wing Span"=pointsInBounds()$`Wing Span`,"Total Body Length"=pointsInBounds()$`Total Body Length`,
                  "Fore Wing Area"=pointsInBounds()$`Fore Wing Area`,"Hind Wing Area"=pointsInBounds()$`Hind Wing Area`,"Antenna"=pointsInBounds()$Antenna,
                  "Antennules"=pointsInBounds()$`Antennules`,"BP (mb) Low"=pointsInBounds()$`BP (mb) Low`,"BP (mb) High"=pointsInBounds()$`BP (mb) High`,
                  "Wind Low (Km/hr)"=pointsInBounds()$`Wind Low (Km/hr)`,"Wind High (Km/hr)"=pointsInBounds()$`Wind High (Km/hr)`,
                  "Temp C Low"=pointsInBounds()$`Temp C Low`,"Temp C High"=pointsInBounds()$`Temp C High`,
                  "Percipitation (mm)"=pointsInBounds()$`Percipitation (mm)`,"Humidity High"=pointsInBounds()$`Humidity High`,
                  "Humidity Low"=pointsInBounds()$`Humidity Low`,"Body Length 1 mm"=pointsInBounds()$`Body Length 1 mm`,
                  "Body Length 2 mm"=pointsInBounds()$`Body Length 2 mm`,"Body Length 3 mm"=pointsInBounds()$`Body Length 3 mm`,
                  "Fore Wing Length01mm"=pointsInBounds()$`Fore Wing Length01mm`,
                  "Fore Wing Length02mm"=pointsInBounds()$`Fore Wing Length02mm`,"Fore Wing Width01mm"=pointsInBounds()$`Fore Wing Width01mm`,
                  "Fore Wing Width02mm"=pointsInBounds()$`Fore Wing Width02mm`,"Fore Wing Width03mm"=pointsInBounds()$`Fore Wing Width03mm`,
                  "Fore Wing Width04mm"=pointsInBounds()$`Fore Wing Width04mm`,"Fore Wing Width05mm"=pointsInBounds()$`Fore Wing Width05mm`,
                  "Fore Wing Width06mm"=pointsInBounds()$`Fore Wing Width06mm`,"Fore Wing Width07mm"=pointsInBounds()$`Fore Wing Width07mm`,
                  "Hind Wing Length01mm"=pointsInBounds()$`Hind Wing Length01mm`,"HWL02mm"=pointsInBounds()$`HWL02mm`,
                  "Hind Wing Width01mm"=pointsInBounds()$`Hind Wing Width01mm`,"Hind Wing Width02mm"=pointsInBounds()$`Hind Wing Width02mm`,
                  "Hind Wing Width03mm"=pointsInBounds()$`Hind Wing Width03mm`,"Hind Wing Width04mm"=pointsInBounds()$`Hind Wing Width04mm`,
                  "Hind Wing Width05mm"=pointsInBounds()$`Hind Wing Width05mm`,"Hind Wing Width06mm"=pointsInBounds()$`Hind Wing Width06mm`,
                  "Hind Wing Width07mm"=pointsInBounds()$`Hind Wing Width07mm`,"Fore Wing perimeter"=pointsInBounds()$`Fore Wing perimeter`,
                  "Hind Wing perimeter"=pointsInBounds()$`Hind Wing perimeter`,"Body Width01mm"=pointsInBounds()$`Body Width01mm`,
                  "Body Width02mm"=pointsInBounds()$`Body Width02mm`,"Body Width03mm"=pointsInBounds()$`Body Width03mm`,
                  "Telson L"=pointsInBounds()$`Telson L`,"Telson W"=pointsInBounds()$`Telson W`,"Orbital"=pointsInBounds()$`Orbital`,
                  "W. B/t orbitals"=pointsInBounds()$`W. B/t orbitals`,"Chela L"=pointsInBounds()$`Chela L`,"Chela W"=pointsInBounds()$`Chela W`,
                  "chela + immovable finger"=pointsInBounds()$`chela + immovable finger`,"movable finger"=pointsInBounds()$`movable finger`,
                  "Librigena Length"=pointsInBounds()$`Librigena Length`,"Head thickness"=pointsInBounds()$`Head thickness`,
                  "Body Thickness 1"=pointsInBounds()$`Body Thickness 1`,"Body Thickness 2"=pointsInBounds()$`Body Thickness 2`,
                  "Skull Length"=pointsInBounds()$`Skull Length`,"Skull Hieght"=pointsInBounds()$`Skull Hieght`,"Skull Width"=pointsInBounds()$`Skull Width`,
                  "Neck"=pointsInBounds()$`Neck`,"Rib cage Length"=pointsInBounds()$`Rib cage Length`,"Femur"=pointsInBounds()$`Femur`,
                  "Tibia"=pointsInBounds()$`Tibia`,"Foot Length"=pointsInBounds()$`Foot Length`,"Foot Width"=pointsInBounds()$`Foot Width`,
                  "Humerous"=pointsInBounds()$`Humerous`,"Lower arm"=pointsInBounds()$`Lower arm`,"Digits 1"=pointsInBounds()$`Digits 1`,
                  "Digits 2"=pointsInBounds()$`Digits 2`,"Digits 3"=pointsInBounds()$`Digits 3`,"Digits 4"=pointsInBounds()$`Digits 4`,
                  "Tail"=pointsInBounds()$`Tail`),colorByPoint=T,showInLegend=F)
  })
  
  ##Density
  output$Den <- renderPlot({plot1<-
    # If no organisms are in view, don't plot
    if (nrow(pointsInBounds()) == 0)
      return()
  Data[Data==-1]<-NA
  plot(density(na.omit(switch(input$DX,"Wing Span"=pointsInBounds()$`Wing Span`,"Total Body Length"=pointsInBounds()$`Total Body Length`,
                              "Fore Wing Area"=pointsInBounds()$`Fore Wing Area`,"Hind Wing Area"=pointsInBounds()$`Hind Wing Area`,"Antenna"=pointsInBounds()$Antenna,
                              "Antennules"=pointsInBounds()$`Antennules`,"BP (mb) Low"=pointsInBounds()$`BP (mb) Low`,"BP (mb) High"=pointsInBounds()$`BP (mb) High`,
                              "Wind Low (Km/hr)"=pointsInBounds()$`Wind Low (Km/hr)`,"Wind High (Km/hr)"=pointsInBounds()$`Wind High (Km/hr)`,
                              "Temp C Low"=pointsInBounds()$`Temp C Low`,"Temp C High"=pointsInBounds()$`Temp C High`,
                              "Percipitation (mm)"=pointsInBounds()$`Percipitation (mm)`,"Humidity High"=pointsInBounds()$`Humidity High`,
                              "Humidity Low"=pointsInBounds()$`Humidity Low`,"Body Length 1 mm"=pointsInBounds()$`Body Length 1 mm`,
                              "Body Length 2 mm"=pointsInBounds()$`Body Length 2 mm`,"Body Length 3 mm"=pointsInBounds()$`Body Length 3 mm`,
                              "Fore Wing Length01mm"=pointsInBounds()$`Fore Wing Length01mm`,
                              "Fore Wing Length02mm"=pointsInBounds()$`Fore Wing Length02mm`,"Fore Wing Width01mm"=pointsInBounds()$`Fore Wing Width01mm`,
                              "Fore Wing Width02mm"=pointsInBounds()$`Fore Wing Width02mm`,"Fore Wing Width03mm"=pointsInBounds()$`Fore Wing Width03mm`,
                              "Fore Wing Width04mm"=pointsInBounds()$`Fore Wing Width04mm`,"Fore Wing Width05mm"=pointsInBounds()$`Fore Wing Width05mm`,
                              "Fore Wing Width06mm"=pointsInBounds()$`Fore Wing Width06mm`,"Fore Wing Width07mm"=pointsInBounds()$`Fore Wing Width07mm`,
                              "Hind Wing Length01mm"=pointsInBounds()$`Hind Wing Length01mm`,"HWL02mm"=pointsInBounds()$`HWL02mm`,
                              "Hind Wing Width01mm"=pointsInBounds()$`Hind Wing Width01mm`,"Hind Wing Width02mm"=pointsInBounds()$`Hind Wing Width02mm`,
                              "Hind Wing Width03mm"=pointsInBounds()$`Hind Wing Width03mm`,"Hind Wing Width04mm"=pointsInBounds()$`Hind Wing Width04mm`,
                              "Hind Wing Width05mm"=pointsInBounds()$`Hind Wing Width05mm`,"Hind Wing Width06mm"=pointsInBounds()$`Hind Wing Width06mm`,
                              "Hind Wing Width07mm"=pointsInBounds()$`Hind Wing Width07mm`,"Fore Wing perimeter"=pointsInBounds()$`Fore Wing perimeter`,
                              "Hind Wing perimeter"=pointsInBounds()$`Hind Wing perimeter`,"Body Width01mm"=pointsInBounds()$`Body Width01mm`,
                              "Body Width02mm"=pointsInBounds()$`Body Width02mm`,"Body Width03mm"=pointsInBounds()$`Body Width03mm`,
                              "Telson L"=pointsInBounds()$`Telson L`,"Telson W"=pointsInBounds()$`Telson W`,"Orbital"=pointsInBounds()$`Orbital`,
                              "W. B/t orbitals"=pointsInBounds()$`W. B/t orbitals`,"Chela L"=pointsInBounds()$`Chela L`,"Chela W"=pointsInBounds()$`Chela W`,
                              "chela + immovable finger"=pointsInBounds()$`chela + immovable finger`,"movable finger"=pointsInBounds()$`movable finger`,
                              "Librigena Length"=pointsInBounds()$`Librigena Length`,"Head thickness"=pointsInBounds()$`Head thickness`,
                              "Body Thickness 1"=pointsInBounds()$`Body Thickness 1`,"Body Thickness 2"=pointsInBounds()$`Body Thickness 2`,
                              "Skull Length"=pointsInBounds()$`Skull Length`,"Skull Hieght"=pointsInBounds()$`Skull Hieght`,"Skull Width"=pointsInBounds()$`Skull Width`,
                              "Neck"=pointsInBounds()$`Neck`,"Rib cage Length"=pointsInBounds()$`Rib cage Length`,"Femur"=pointsInBounds()$`Femur`,
                              "Tibia"=pointsInBounds()$`Tibia`,"Foot Length"=pointsInBounds()$`Foot Length`,"Foot Width"=pointsInBounds()$`Foot Width`,
                              "Humerous"=pointsInBounds()$`Humerous`,"Lower arm"=pointsInBounds()$`Lower arm`,"Digits 1"=pointsInBounds()$`Digits 1`,
                              "Digits 2"=pointsInBounds()$`Digits 2`,"Digits 3"=pointsInBounds()$`Digits 3`,"Digits 4"=pointsInBounds()$`Digits 4`,
                              "Tail"=pointsInBounds()$`Tail`))),
       main = "",
       col = rainbow(30),
       border = 'transparent')
  })
  
  ##scatterplot
  output$Bokeh<-renderRbokeh({
    if (nrow(pointsInBounds()) == 0)
      return(NULL)
    Data[Data==-1]<-NA
    figure(legend_location = NULL,toolbar_location = "right", tools = c("pan","wheel_zoom","box_zoom","reset","save"))%>%
      ly_points(x=switch(input$BX,"Wing Span"=pointsInBounds()$`Wing Span`,"Total Body Length"=pointsInBounds()$`Total Body Length`,
                         "Fore Wing Area"=pointsInBounds()$`Fore Wing Area`,"Hind Wing Area"=pointsInBounds()$`Hind Wing Area`,"Antenna"=pointsInBounds()$Antenna,
                         "Antennules"=pointsInBounds()$`Antennules`,"BP (mb) Low"=pointsInBounds()$`BP (mb) Low`,"BP (mb) High"=pointsInBounds()$`BP (mb) High`,
                         "Wind Low (Km/hr)"=pointsInBounds()$`Wind Low (Km/hr)`,"Wind High (Km/hr)"=pointsInBounds()$`Wind High (Km/hr)`,
                         "Temp C Low"=pointsInBounds()$`Temp C Low`,"Temp C High"=pointsInBounds()$`Temp C High`,
                         "Percipitation (mm)"=pointsInBounds()$`Percipitation (mm)`,"Humidity High"=pointsInBounds()$`Humidity High`,
                         "Humidity Low"=pointsInBounds()$`Humidity Low`,"Body Length 1 mm"=pointsInBounds()$`Body Length 1 mm`,
                         "Body Length 2 mm"=pointsInBounds()$`Body Length 2 mm`,"Body Length 3 mm"=pointsInBounds()$`Body Length 3 mm`,
                         "Fore Wing Length01mm"=pointsInBounds()$`Fore Wing Length01mm`,
                         "Fore Wing Length02mm"=pointsInBounds()$`Fore Wing Length02mm`,"Fore Wing Width01mm"=pointsInBounds()$`Fore Wing Width01mm`,
                         "Fore Wing Width02mm"=pointsInBounds()$`Fore Wing Width02mm`,"Fore Wing Width03mm"=pointsInBounds()$`Fore Wing Width03mm`,
                         "Fore Wing Width04mm"=pointsInBounds()$`Fore Wing Width04mm`,"Fore Wing Width05mm"=pointsInBounds()$`Fore Wing Width05mm`,
                         "Fore Wing Width06mm"=pointsInBounds()$`Fore Wing Width06mm`,"Fore Wing Width07mm"=pointsInBounds()$`Fore Wing Width07mm`,
                         "Hind Wing Length01mm"=pointsInBounds()$`Hind Wing Length01mm`,"HWL02mm"=pointsInBounds()$`HWL02mm`,
                         "Hind Wing Width01mm"=pointsInBounds()$`Hind Wing Width01mm`,"Hind Wing Width02mm"=pointsInBounds()$`Hind Wing Width02mm`,
                         "Hind Wing Width03mm"=pointsInBounds()$`Hind Wing Width03mm`,"Hind Wing Width04mm"=pointsInBounds()$`Hind Wing Width04mm`,
                         "Hind Wing Width05mm"=pointsInBounds()$`Hind Wing Width05mm`,"Hind Wing Width06mm"=pointsInBounds()$`Hind Wing Width06mm`,
                         "Hind Wing Width07mm"=pointsInBounds()$`Hind Wing Width07mm`,"Fore Wing perimeter"=pointsInBounds()$`Fore Wing perimeter`,
                         "Hind Wing perimeter"=pointsInBounds()$`Hind Wing perimeter`,"Body Width01mm"=pointsInBounds()$`Body Width01mm`,
                         "Body Width02mm"=pointsInBounds()$`Body Width02mm`,"Body Width03mm"=pointsInBounds()$`Body Width03mm`,
                         "Telson L"=pointsInBounds()$`Telson L`,"Telson W"=pointsInBounds()$`Telson W`,"Orbital"=pointsInBounds()$`Orbital`,
                         "W. B/t orbitals"=pointsInBounds()$`W. B/t orbitals`,"Chela L"=pointsInBounds()$`Chela L`,"Chela W"=pointsInBounds()$`Chela W`,
                         "chela + immovable finger"=pointsInBounds()$`chela + immovable finger`,"movable finger"=pointsInBounds()$`movable finger`,
                         "Librigena Length"=pointsInBounds()$`Librigena Length`,"Head thickness"=pointsInBounds()$`Head thickness`,
                         "Body Thickness 1"=pointsInBounds()$`Body Thickness 1`,"Body Thickness 2"=pointsInBounds()$`Body Thickness 2`,
                         "Skull Length"=pointsInBounds()$`Skull Length`,"Skull Hieght"=pointsInBounds()$`Skull Hieght`,"Skull Width"=pointsInBounds()$`Skull Width`,
                         "Neck"=pointsInBounds()$`Neck`,"Rib cage Length"=pointsInBounds()$`Rib cage Length`,"Femur"=pointsInBounds()$`Femur`,
                         "Tibia"=pointsInBounds()$`Tibia`,"Foot Length"=pointsInBounds()$`Foot Length`,"Foot Width"=pointsInBounds()$`Foot Width`,
                         "Humerous"=pointsInBounds()$`Humerous`,"Lower arm"=pointsInBounds()$`Lower arm`,"Digits 1"=pointsInBounds()$`Digits 1`,
                         "Digits 2"=pointsInBounds()$`Digits 2`,"Digits 3"=pointsInBounds()$`Digits 3`,"Digits 4"=pointsInBounds()$`Digits 4`,
                         "Tail"=pointsInBounds()$`Tail`),
                y=switch(input$BY,"Wing Span"=pointsInBounds()$`Wing Span`,"Total Body Length"=pointsInBounds()$`Total Body Length`,
                         "Fore Wing Area"=pointsInBounds()$`Fore Wing Area`,"Hind Wing Area"=pointsInBounds()$`Hind Wing Area`,"Antenna"=pointsInBounds()$Antenna,
                         "Antennules"=pointsInBounds()$`Antennules`,"BP (mb) Low"=pointsInBounds()$`BP (mb) Low`,"BP (mb) High"=pointsInBounds()$`BP (mb) High`,
                         "Wind Low (Km/hr)"=pointsInBounds()$`Wind Low (Km/hr)`,"Wind High (Km/hr)"=pointsInBounds()$`Wind High (Km/hr)`,
                         "Temp C Low"=pointsInBounds()$`Temp C Low`,"Temp C High"=pointsInBounds()$`Temp C High`,
                         "Percipitation (mm)"=pointsInBounds()$`Percipitation (mm)`,"Humidity High"=pointsInBounds()$`Humidity High`,
                         "Humidity Low"=pointsInBounds()$`Humidity Low`,"Body Length 1 mm"=pointsInBounds()$`Body Length 1 mm`,
                         "Body Length 2 mm"=pointsInBounds()$`Body Length 2 mm`,"Body Length 3 mm"=pointsInBounds()$`Body Length 3 mm`,
                         "Fore Wing Length01mm"=pointsInBounds()$`Fore Wing Length01mm`,
                         "Fore Wing Length02mm"=pointsInBounds()$`Fore Wing Length02mm`,"Fore Wing Width01mm"=pointsInBounds()$`Fore Wing Width01mm`,
                         "Fore Wing Width02mm"=pointsInBounds()$`Fore Wing Width02mm`,"Fore Wing Width03mm"=pointsInBounds()$`Fore Wing Width03mm`,
                         "Fore Wing Width04mm"=pointsInBounds()$`Fore Wing Width04mm`,"Fore Wing Width05mm"=pointsInBounds()$`Fore Wing Width05mm`,
                         "Fore Wing Width06mm"=pointsInBounds()$`Fore Wing Width06mm`,"Fore Wing Width07mm"=pointsInBounds()$`Fore Wing Width07mm`,
                         "Hind Wing Length01mm"=pointsInBounds()$`Hind Wing Length01mm`,"HWL02mm"=pointsInBounds()$`HWL02mm`,
                         "Hind Wing Width01mm"=pointsInBounds()$`Hind Wing Width01mm`,"Hind Wing Width02mm"=pointsInBounds()$`Hind Wing Width02mm`,
                         "Hind Wing Width03mm"=pointsInBounds()$`Hind Wing Width03mm`,"Hind Wing Width04mm"=pointsInBounds()$`Hind Wing Width04mm`,
                         "Hind Wing Width05mm"=pointsInBounds()$`Hind Wing Width05mm`,"Hind Wing Width06mm"=pointsInBounds()$`Hind Wing Width06mm`,
                         "Hind Wing Width07mm"=pointsInBounds()$`Hind Wing Width07mm`,"Fore Wing perimeter"=pointsInBounds()$`Fore Wing perimeter`,
                         "Hind Wing perimeter"=pointsInBounds()$`Hind Wing perimeter`,"Body Width01mm"=pointsInBounds()$`Body Width01mm`,
                         "Body Width02mm"=pointsInBounds()$`Body Width02mm`,"Body Width03mm"=pointsInBounds()$`Body Width03mm`,
                         "Telson L"=pointsInBounds()$`Telson L`,"Telson W"=pointsInBounds()$`Telson W`,"Orbital"=pointsInBounds()$`Orbital`,
                         "W. B/t orbitals"=pointsInBounds()$`W. B/t orbitals`,"Chela L"=pointsInBounds()$`Chela L`,"Chela W"=pointsInBounds()$`Chela W`,
                         "chela + immovable finger"=pointsInBounds()$`chela + immovable finger`,"movable finger"=pointsInBounds()$`movable finger`,
                         "Librigena Length"=pointsInBounds()$`Librigena Length`,"Head thickness"=pointsInBounds()$`Head thickness`,
                         "Body Thickness 1"=pointsInBounds()$`Body Thickness 1`,"Body Thickness 2"=pointsInBounds()$`Body Thickness 2`,
                         "Skull Length"=pointsInBounds()$`Skull Length`,"Skull Hieght"=pointsInBounds()$`Skull Hieght`,"Skull Width"=pointsInBounds()$`Skull Width`,
                         "Neck"=pointsInBounds()$`Neck`,"Rib cage Length"=pointsInBounds()$`Rib cage Length`,"Femur"=pointsInBounds()$`Femur`,
                         "Tibia"=pointsInBounds()$`Tibia`,"Foot Length"=pointsInBounds()$`Foot Length`,"Foot Width"=pointsInBounds()$`Foot Width`,
                         "Humerous"=pointsInBounds()$`Humerous`,"Lower arm"=pointsInBounds()$`Lower arm`,"Digits 1"=pointsInBounds()$`Digits 1`,
                         "Digits 2"=pointsInBounds()$`Digits 2`,"Digits 3"=pointsInBounds()$`Digits 3`,"Digits 4"=pointsInBounds()$`Digits 4`,
                         "Tail"=pointsInBounds()$`Tail`),
                data = pointsInBounds(),
                xlab=switch(input$BX,"Wing Span"="Wing Span","Total Body Length"="Total Body Length","Fore Wing Area"="Fore Wing Area",
                            "Hind Wing Area"="Hind Wing Area","Antenna"="Antenna","Antennules"="Antennules","Temp C Low"="Temp C Low","BP (mb) Low"="BP (mb) Low","BP (mb) High"="BP (mb) High",
                            "Wind Low (Km/hr)"="Wind Low (Km/hr)","Wind High (Km/hr)"="Wind High (Km/hr)","Temp C High"="Temp C High","Percipitation (mm)"="Percipitation (mm)",
                            "Humidity High"="Humidity High","Humidity Low"="Humidity Low","Body Length 1 mm"="Body Length 1 mm","Body Length 2 mm"="Body Length 2 mm",
                            "Body Length 3 mm"="Body Length 3 mm","Fore Wing Length01mm"="Fore Wing Length01mm",
                            "Fore Wing Length02mm"="Fore Wing Length02mm","Fore Wing Width01mm"="Fore Wing Width01mm","Fore Wing Width02mm"="Fore Wing Width02mm",
                            "Fore Wing Width03mm"="Fore Wing Width03mm","Fore Wing Width04mm"="Fore Wing Width04mm","Fore Wing Width05mm"="Fore Wing Width05mm",
                            "Fore Wing Width06mm"="Fore Wing Width06mm","Fore Wing Width07mm"="Fore Wing Width07mm","Hind Wing Length01mm"="Hind Wing Length01mm",
                            "HWL02mm"="HWL02mm","Hind Wing Width01mm"="Hind Wing Width01mm","Hind Wing Width02mm"="Hind Wing Width02mm","Hind Wing Width03mm"="Hind Wing Width03mm",
                            "Hind Wing Width04mm"="Hind Wing Width04mm","Hind Wing Width05mm"="Hind Wing Width05mm","Hind Wing Width06mm"="Hind Wing Width06mm",
                            "Hind Wing Width07mm"="Hind Wing Width07mm","Hind Wing perimeter"="Hind Wing perimeter","Hind Wing perimeter"="Hind Wing perimeter",
                            "Body Width01mm"="Body Width01mm","Body Width02mm"="Body Width02mm","Body Width03mm"="Body Width03mm",
                            "Telson L"="Telson L","Telson W"="Telson W","Orbital"="Orbital","W. B/t orbitals"="W. B/t orbitals","Chela L"="Chela L","Chela W"="Chela W",
                            "chela + immovable finger"="chela + immovable finger","movable finger"="movable finger",
                            "Librigena Length"="Librigena Length","Head thickness"="Head thickness","Body Thickness 1"="Body Thickness 1","Body Thickness 2"="Body Thickness 2",
                            "Skull Length"="Skull Length","Skull Hieght"="Skull Hieght","Skull Width"="Skull Width","Neck"="Neck","Rib cage Length"="Rib cage Length",
                            "Femur"="Femur","Tibia"="Tibia","Foot Length"="Foot Length","Foot Width"="Foot Width","Humerous"="Humerous","Lower arm"="Lower arm",
                            "Digits 1"="Digits 1","Digits 2"="Digits 2","Digits 3"="Digits 3","Digits 4"="Digits 4","Tail"="Tail"),
                ylab = switch(input$BY,"Wing Span"="Wing Span","Total Body Length"="Total Body Length","Fore Wing Area"="Fore Wing Area",
                              "Hind Wing Area"="Hind Wing Area","Antenna"="Antenna","Antennules"="Antennules","Temp C Low"="Temp C Low","BP (mb) Low"="BP (mb) Low","BP (mb) High"="BP (mb) High",
                              "Wind Low (Km/hr)"="Wind Low (Km/hr)","Wind High (Km/hr)"="Wind High (Km/hr)","Temp C High"="Temp C High","Percipitation (mm)"="Percipitation (mm)",
                              "Humidity High"="Humidity High","Humidity Low"="Humidity Low","Body Length 1 mm"="Body Length 1 mm","Body Length 2 mm"="Body Length 2 mm",
                              "Body Length 3 mm"="Body Length 3 mm","Fore Wing Length01mm"="Fore Wing Length01mm",
                              "Fore Wing Length02mm"="Fore Wing Length02mm","Fore Wing Width01mm"="Fore Wing Width01mm","Fore Wing Width02mm"="Fore Wing Width02mm",
                              "Fore Wing Width03mm"="Fore Wing Width03mm","Fore Wing Width04mm"="Fore Wing Width04mm","Fore Wing Width05mm"="Fore Wing Width05mm",
                              "Fore Wing Width06mm"="Fore Wing Width06mm","Fore Wing Width07mm"="Fore Wing Width07mm","Hind Wing Length01mm"="Hind Wing Length01mm",
                              "HWL02mm"="HWL02mm","Hind Wing Width01mm"="Hind Wing Width01mm","Hind Wing Width02mm"="Hind Wing Width02mm","Hind Wing Width03mm"="Hind Wing Width03mm",
                              "Hind Wing Width04mm"="Hind Wing Width04mm","Hind Wing Width05mm"="Hind Wing Width05mm","Hind Wing Width06mm"="Hind Wing Width06mm",
                              "Hind Wing Width07mm"="Hind Wing Width07mm","Hind Wing perimeter"="Hind Wing perimeter","Hind Wing perimeter"="Hind Wing perimeter",
                              "Body Width01mm"="Body Width01mm","Body Width02mm"="Body Width02mm","Body Width03mm"="Body Width03mm",
                              "Telson L"="Telson L","Telson W"="Telson W","Orbital"="Orbital","W. B/t orbitals"="W. B/t orbitals","Chela L"="Chela L","Chela W"="Chela W",
                              "chela + immovable finger"="chela + immovable finger","movable finger"="movable finger",
                              "Librigena Length"="Librigena Length","Head thickness"="Head thickness","Body Thickness 1"="Body Thickness 1","Body Thickness 2"="Body Thickness 2",
                              "Skull Length"="Skull Length","Skull Hieght"="Skull Hieght","Skull Width"="Skull Width","Neck"="Neck","Rib cage Length"="Rib cage Length",
                              "Femur"="Femur","Tibia"="Tibia","Foot Length"="Foot Length","Foot Width"="Foot Width","Humerous"="Humerous","Lower arm"="Lower arm",
                              "Digits 1"="Digits 1","Digits 2"="Digits 2","Digits 3"="Digits 3","Digits 4"="Digits 4","Tail"="Tail"),
                color = Species, glyph = Species,hover = c(Specimen,Family))
  })
  
  ##3D scatterplot
  output$threeD<-renderPlotly({
    if (nrow(pointsInBounds()) == 0)
      return(NULL)
    Data[Data==-1]<-NA
    plot_ly(type = "scatter3d",x=switch(input$TX,"Wing Span"=pointsInBounds()$`Wing Span`,"Total Body Length"=pointsInBounds()$`Total Body Length`,
                                        "Fore Wing Area"=pointsInBounds()$`Fore Wing Area`,"Hind Wing Area"=pointsInBounds()$`Hind Wing Area`,"Antenna"=pointsInBounds()$Antenna,
                                        "Antennules"=pointsInBounds()$`Antennules`,"BP (mb) Low"=pointsInBounds()$`BP (mb) Low`,"BP (mb) High"=pointsInBounds()$`BP (mb) High`,
                                        "Wind Low (Km/hr)"=pointsInBounds()$`Wind Low (Km/hr)`,"Wind High (Km/hr)"=pointsInBounds()$`Wind High (Km/hr)`,
                                        "Temp C Low"=pointsInBounds()$`Temp C Low`,"Temp C High"=pointsInBounds()$`Temp C High`,
                                        "Percipitation (mm)"=pointsInBounds()$`Percipitation (mm)`,"Humidity High"=pointsInBounds()$`Humidity High`,
                                        "Humidity Low"=pointsInBounds()$`Humidity Low`,"Body Length 1 mm"=pointsInBounds()$`Body Length 1 mm`,
                                        "Body Length 2 mm"=pointsInBounds()$`Body Length 2 mm`,"Body Length 3 mm"=pointsInBounds()$`Body Length 3 mm`,
                                        "Fore Wing Length01mm"=pointsInBounds()$`Fore Wing Length01mm`,
                                        "Fore Wing Length02mm"=pointsInBounds()$`Fore Wing Length02mm`,"Fore Wing Width01mm"=pointsInBounds()$`Fore Wing Width01mm`,
                                        "Fore Wing Width02mm"=pointsInBounds()$`Fore Wing Width02mm`,"Fore Wing Width03mm"=pointsInBounds()$`Fore Wing Width03mm`,
                                        "Fore Wing Width04mm"=pointsInBounds()$`Fore Wing Width04mm`,"Fore Wing Width05mm"=pointsInBounds()$`Fore Wing Width05mm`,
                                        "Fore Wing Width06mm"=pointsInBounds()$`Fore Wing Width06mm`,"Fore Wing Width07mm"=pointsInBounds()$`Fore Wing Width07mm`,
                                        "Hind Wing Length01mm"=pointsInBounds()$`Hind Wing Length01mm`,"HWL02mm"=pointsInBounds()$`HWL02mm`,
                                        "Hind Wing Width01mm"=pointsInBounds()$`Hind Wing Width01mm`,"Hind Wing Width02mm"=pointsInBounds()$`Hind Wing Width02mm`,
                                        "Hind Wing Width03mm"=pointsInBounds()$`Hind Wing Width03mm`,"Hind Wing Width04mm"=pointsInBounds()$`Hind Wing Width04mm`,
                                        "Hind Wing Width05mm"=pointsInBounds()$`Hind Wing Width05mm`,"Hind Wing Width06mm"=pointsInBounds()$`Hind Wing Width06mm`,
                                        "Hind Wing Width07mm"=pointsInBounds()$`Hind Wing Width07mm`,"Fore Wing perimeter"=pointsInBounds()$`Fore Wing perimeter`,
                                        "Hind Wing perimeter"=pointsInBounds()$`Hind Wing perimeter`,"Body Width01mm"=pointsInBounds()$`Body Width01mm`,
                                        "Body Width02mm"=pointsInBounds()$`Body Width02mm`,"Body Width03mm"=pointsInBounds()$`Body Width03mm`,
                                        "Telson L"=pointsInBounds()$`Telson L`,"Telson W"=pointsInBounds()$`Telson W`,"Orbital"=pointsInBounds()$`Orbital`,
                                        "W. B/t orbitals"=pointsInBounds()$`W. B/t orbitals`,"Chela L"=pointsInBounds()$`Chela L`,"Chela W"=pointsInBounds()$`Chela W`,
                                        "chela + immovable finger"=pointsInBounds()$`chela + immovable finger`,"movable finger"=pointsInBounds()$`movable finger`,
                                        "Librigena Length"=pointsInBounds()$`Librigena Length`,"Head thickness"=pointsInBounds()$`Head thickness`,
                                        "Body Thickness 1"=pointsInBounds()$`Body Thickness 1`,"Body Thickness 2"=pointsInBounds()$`Body Thickness 2`,
                                        "Skull Length"=pointsInBounds()$`Skull Length`,"Skull Hieght"=pointsInBounds()$`Skull Hieght`,"Skull Width"=pointsInBounds()$`Skull Width`,
                                        "Neck"=pointsInBounds()$`Neck`,"Rib cage Length"=pointsInBounds()$`Rib cage Length`,"Femur"=pointsInBounds()$`Femur`,
                                        "Tibia"=pointsInBounds()$`Tibia`,"Foot Length"=pointsInBounds()$`Foot Length`,"Foot Width"=pointsInBounds()$`Foot Width`,
                                        "Humerous"=pointsInBounds()$`Humerous`,"Lower arm"=pointsInBounds()$`Lower arm`,"Digits 1"=pointsInBounds()$`Digits 1`,
                                        "Digits 2"=pointsInBounds()$`Digits 2`,"Digits 3"=pointsInBounds()$`Digits 3`,"Digits 4"=pointsInBounds()$`Digits 4`,
                                        "Tail"=pointsInBounds()$`Tail`),
            y=switch(input$TY,"Wing Span"=pointsInBounds()$`Wing Span`,"Total Body Length"=pointsInBounds()$`Total Body Length`,
                     "Fore Wing Area"=pointsInBounds()$`Fore Wing Area`,"Hind Wing Area"=pointsInBounds()$`Hind Wing Area`,"Antenna"=pointsInBounds()$Antenna,
                     "Antennules"=pointsInBounds()$`Antennules`,"BP (mb) Low"=pointsInBounds()$`BP (mb) Low`,"BP (mb) High"=pointsInBounds()$`BP (mb) High`,
                     "Wind Low (Km/hr)"=pointsInBounds()$`Wind Low (Km/hr)`,"Wind High (Km/hr)"=pointsInBounds()$`Wind High (Km/hr)`,
                     "Temp C Low"=pointsInBounds()$`Temp C Low`,"Temp C High"=pointsInBounds()$`Temp C High`,
                     "Percipitation (mm)"=pointsInBounds()$`Percipitation (mm)`,"Humidity High"=pointsInBounds()$`Humidity High`,
                     "Humidity Low"=pointsInBounds()$`Humidity Low`,"Body Length 1 mm"=pointsInBounds()$`Body Length 1 mm`,
                     "Body Length 2 mm"=pointsInBounds()$`Body Length 2 mm`,"Body Length 3 mm"=pointsInBounds()$`Body Length 3 mm`,
                     "Fore Wing Length01mm"=pointsInBounds()$`Fore Wing Length01mm`,
                     "Fore Wing Length02mm"=pointsInBounds()$`Fore Wing Length02mm`,"Fore Wing Width01mm"=pointsInBounds()$`Fore Wing Width01mm`,
                     "Fore Wing Width02mm"=pointsInBounds()$`Fore Wing Width02mm`,"Fore Wing Width03mm"=pointsInBounds()$`Fore Wing Width03mm`,
                     "Fore Wing Width04mm"=pointsInBounds()$`Fore Wing Width04mm`,"Fore Wing Width05mm"=pointsInBounds()$`Fore Wing Width05mm`,
                     "Fore Wing Width06mm"=pointsInBounds()$`Fore Wing Width06mm`,"Fore Wing Width07mm"=pointsInBounds()$`Fore Wing Width07mm`,
                     "Hind Wing Length01mm"=pointsInBounds()$`Hind Wing Length01mm`,"HWL02mm"=pointsInBounds()$`HWL02mm`,
                     "Hind Wing Width01mm"=pointsInBounds()$`Hind Wing Width01mm`,"Hind Wing Width02mm"=pointsInBounds()$`Hind Wing Width02mm`,
                     "Hind Wing Width03mm"=pointsInBounds()$`Hind Wing Width03mm`,"Hind Wing Width04mm"=pointsInBounds()$`Hind Wing Width04mm`,
                     "Hind Wing Width05mm"=pointsInBounds()$`Hind Wing Width05mm`,"Hind Wing Width06mm"=pointsInBounds()$`Hind Wing Width06mm`,
                     "Hind Wing Width07mm"=pointsInBounds()$`Hind Wing Width07mm`,"Fore Wing perimeter"=pointsInBounds()$`Fore Wing perimeter`,
                     "Hind Wing perimeter"=pointsInBounds()$`Hind Wing perimeter`,"Body Width01mm"=pointsInBounds()$`Body Width01mm`,
                     "Body Width02mm"=pointsInBounds()$`Body Width02mm`,"Body Width03mm"=pointsInBounds()$`Body Width03mm`,
                     "Telson L"=pointsInBounds()$`Telson L`,"Telson W"=pointsInBounds()$`Telson W`,"Orbital"=pointsInBounds()$`Orbital`,
                     "W. B/t orbitals"=pointsInBounds()$`W. B/t orbitals`,"Chela L"=pointsInBounds()$`Chela L`,"Chela W"=pointsInBounds()$`Chela W`,
                     "chela + immovable finger"=pointsInBounds()$`chela + immovable finger`,"movable finger"=pointsInBounds()$`movable finger`,
                     "Librigena Length"=pointsInBounds()$`Librigena Length`,"Head thickness"=pointsInBounds()$`Head thickness`,
                     "Body Thickness 1"=pointsInBounds()$`Body Thickness 1`,"Body Thickness 2"=pointsInBounds()$`Body Thickness 2`,
                     "Skull Length"=pointsInBounds()$`Skull Length`,"Skull Hieght"=pointsInBounds()$`Skull Hieght`,"Skull Width"=pointsInBounds()$`Skull Width`,
                     "Neck"=pointsInBounds()$`Neck`,"Rib cage Length"=pointsInBounds()$`Rib cage Length`,"Femur"=pointsInBounds()$`Femur`,
                     "Tibia"=pointsInBounds()$`Tibia`,"Foot Length"=pointsInBounds()$`Foot Length`,"Foot Width"=pointsInBounds()$`Foot Width`,
                     "Humerous"=pointsInBounds()$`Humerous`,"Lower arm"=pointsInBounds()$`Lower arm`,"Digits 1"=pointsInBounds()$`Digits 1`,
                     "Digits 2"=pointsInBounds()$`Digits 2`,"Digits 3"=pointsInBounds()$`Digits 3`,"Digits 4"=pointsInBounds()$`Digits 4`,
                     "Tail"=pointsInBounds()$`Tail`),
            z=switch(input$TZ,"Wing Span"=pointsInBounds()$`Wing Span`,"Total Body Length"=pointsInBounds()$`Total Body Length`,
                     "Fore Wing Area"=pointsInBounds()$`Fore Wing Area`,"Hind Wing Area"=pointsInBounds()$`Hind Wing Area`,"Antenna"=pointsInBounds()$Antenna,
                     "Antennules"=pointsInBounds()$`Antennules`,"BP (mb) Low"=pointsInBounds()$`BP (mb) Low`,"BP (mb) High"=pointsInBounds()$`BP (mb) High`,
                     "Wind Low (Km/hr)"=pointsInBounds()$`Wind Low (Km/hr)`,"Wind High (Km/hr)"=pointsInBounds()$`Wind High (Km/hr)`,
                     "Temp C Low"=pointsInBounds()$`Temp C Low`,"Temp C High"=pointsInBounds()$`Temp C High`,
                     "Percipitation (mm)"=pointsInBounds()$`Percipitation (mm)`,"Humidity High"=pointsInBounds()$`Humidity High`,
                     "Humidity Low"=pointsInBounds()$`Humidity Low`,"Body Length 1 mm"=pointsInBounds()$`Body Length 1 mm`,
                     "Body Length 2 mm"=pointsInBounds()$`Body Length 2 mm`,"Body Length 3 mm"=pointsInBounds()$`Body Length 3 mm`,
                     "Fore Wing Length01mm"=pointsInBounds()$`Fore Wing Length01mm`,
                     "Fore Wing Length02mm"=pointsInBounds()$`Fore Wing Length02mm`,"Fore Wing Width01mm"=pointsInBounds()$`Fore Wing Width01mm`,
                     "Fore Wing Width02mm"=pointsInBounds()$`Fore Wing Width02mm`,"Fore Wing Width03mm"=pointsInBounds()$`Fore Wing Width03mm`,
                     "Fore Wing Width04mm"=pointsInBounds()$`Fore Wing Width04mm`,"Fore Wing Width05mm"=pointsInBounds()$`Fore Wing Width05mm`,
                     "Fore Wing Width06mm"=pointsInBounds()$`Fore Wing Width06mm`,"Fore Wing Width07mm"=pointsInBounds()$`Fore Wing Width07mm`,
                     "Hind Wing Length01mm"=pointsInBounds()$`Hind Wing Length01mm`,"HWL02mm"=pointsInBounds()$`HWL02mm`,
                     "Hind Wing Width01mm"=pointsInBounds()$`Hind Wing Width01mm`,"Hind Wing Width02mm"=pointsInBounds()$`Hind Wing Width02mm`,
                     "Hind Wing Width03mm"=pointsInBounds()$`Hind Wing Width03mm`,"Hind Wing Width04mm"=pointsInBounds()$`Hind Wing Width04mm`,
                     "Hind Wing Width05mm"=pointsInBounds()$`Hind Wing Width05mm`,"Hind Wing Width06mm"=pointsInBounds()$`Hind Wing Width06mm`,
                     "Hind Wing Width07mm"=pointsInBounds()$`Hind Wing Width07mm`,"Fore Wing perimeter"=pointsInBounds()$`Fore Wing perimeter`,
                     "Hind Wing perimeter"=pointsInBounds()$`Hind Wing perimeter`,"Body Width01mm"=pointsInBounds()$`Body Width01mm`,
                     "Body Width02mm"=pointsInBounds()$`Body Width02mm`,"Body Width03mm"=pointsInBounds()$`Body Width03mm`,
                     "Telson L"=pointsInBounds()$`Telson L`,"Telson W"=pointsInBounds()$`Telson W`,"Orbital"=pointsInBounds()$`Orbital`,
                     "W. B/t orbitals"=pointsInBounds()$`W. B/t orbitals`,"Chela L"=pointsInBounds()$`Chela L`,"Chela W"=pointsInBounds()$`Chela W`,
                     "chela + immovable finger"=pointsInBounds()$`chela + immovable finger`,"movable finger"=pointsInBounds()$`movable finger`,
                     "Librigena Length"=pointsInBounds()$`Librigena Length`,"Head thickness"=pointsInBounds()$`Head thickness`,
                     "Body Thickness 1"=pointsInBounds()$`Body Thickness 1`,"Body Thickness 2"=pointsInBounds()$`Body Thickness 2`,
                     "Skull Length"=pointsInBounds()$`Skull Length`,"Skull Hieght"=pointsInBounds()$`Skull Hieght`,"Skull Width"=pointsInBounds()$`Skull Width`,
                     "Neck"=pointsInBounds()$`Neck`,"Rib cage Length"=pointsInBounds()$`Rib cage Length`,"Femur"=pointsInBounds()$`Femur`,
                     "Tibia"=pointsInBounds()$`Tibia`,"Foot Length"=pointsInBounds()$`Foot Length`,"Foot Width"=pointsInBounds()$`Foot Width`,
                     "Humerous"=pointsInBounds()$`Humerous`,"Lower arm"=pointsInBounds()$`Lower arm`,"Digits 1"=pointsInBounds()$`Digits 1`,
                     "Digits 2"=pointsInBounds()$`Digits 2`,"Digits 3"=pointsInBounds()$`Digits 3`,"Digits 4"=pointsInBounds()$`Digits 4`,
                     "Tail"=pointsInBounds()$`Tail`),
            mode = 'markers',marker = list(symbol="circle-open",size = 5),color = pointsInBounds()$Species, colors=rainbow(500),textposition="top center",
            text = ~paste('Order:',pointsInBounds()$Order,'<br>Family:',pointsInBounds()$Family,'<br>',pointsInBounds()$Specimen),showlegend=FALSE,hoverinfo=switch(input$pop,"Info"="text","numbers"="x+y+z"))%>%
      add_markers() %>%
      hide_colorbar()%>%
      config(displayModeBar = "hover", workspace = TRUE, sendData = FALSE, displaylogo = FALSE)%>%
      layout(margin=list(l = 0,r = 0, b = 0,t = 0,pad = 0),scene = list(
        xaxis = list(title = switch(input$TX,"Wing Span"="Wing Span","Total Body Length"="Total Body Length","Fore Wing Area"="Fore Wing Area",
                                    "Hind Wing Area"="Hind Wing Area","Antenna"="Antenna","Antennules"="Antennules","Temp C Low"="Temp C Low","BP (mb) Low"="BP (mb) Low","BP (mb) High"="BP (mb) High",
                                    "Wind Low (Km/hr)"="Wind Low (Km/hr)","Wind High (Km/hr)"="Wind High (Km/hr)","Temp C High"="Temp C High","Percipitation (mm)"="Percipitation (mm)",
                                    "Humidity High"="Humidity High","Humidity Low"="Humidity Low","Body Length 1 mm"="Body Length 1 mm","Body Length 2 mm"="Body Length 2 mm",
                                    "Body Length 3 mm"="Body Length 3 mm","Fore Wing Length01mm"="Fore Wing Length01mm",
                                    "Fore Wing Length02mm"="Fore Wing Length02mm","Fore Wing Width01mm"="Fore Wing Width01mm","Fore Wing Width02mm"="Fore Wing Width02mm",
                                    "Fore Wing Width03mm"="Fore Wing Width03mm","Fore Wing Width04mm"="Fore Wing Width04mm","Fore Wing Width05mm"="Fore Wing Width05mm",
                                    "Fore Wing Width06mm"="Fore Wing Width06mm","Fore Wing Width07mm"="Fore Wing Width07mm","Hind Wing Length01mm"="Hind Wing Length01mm",
                                    "HWL02mm"="HWL02mm","Hind Wing Width01mm"="Hind Wing Width01mm","Hind Wing Width02mm"="Hind Wing Width02mm","Hind Wing Width03mm"="Hind Wing Width03mm",
                                    "Hind Wing Width04mm"="Hind Wing Width04mm","Hind Wing Width05mm"="Hind Wing Width05mm","Hind Wing Width06mm"="Hind Wing Width06mm",
                                    "Hind Wing Width07mm"="Hind Wing Width07mm","Hind Wing perimeter"="Hind Wing perimeter","Hind Wing perimeter"="Hind Wing perimeter",
                                    "Body Width01mm"="Body Width01mm","Body Width02mm"="Body Width02mm","Body Width03mm"="Body Width03mm",
                                    "Telson L"="Telson L","Telson W"="Telson W","Orbital"="Orbital","W. B/t orbitals"="W. B/t orbitals","Chela L"="Chela L","Chela W"="Chela W",
                                    "chela + immovable finger"="chela + immovable finger","movable finger"="movable finger",
                                    "Librigena Length"="Librigena Length","Head thickness"="Head thickness","Body Thickness 1"="Body Thickness 1","Body Thickness 2"="Body Thickness 2",
                                    "Skull Length"="Skull Length","Skull Hieght"="Skull Hieght","Skull Width"="Skull Width","Neck"="Neck","Rib cage Length"="Rib cage Length",
                                    "Femur"="Femur","Tibia"="Tibia","Foot Length"="Foot Length","Foot Width"="Foot Width","Humerous"="Humerous","Lower arm"="Lower arm",
                                    "Digits 1"="Digits 1","Digits 2"="Digits 2","Digits 3"="Digits 3","Digits 4"="Digits 4","Tail"="Tail")),
        yaxis = list(title =switch(input$TY,"Wing Span"="Wing Span","Total Body Length"="Total Body Length","Fore Wing Area"="Fore Wing Area",
                                   "Hind Wing Area"="Hind Wing Area","Antenna"="Antenna","Antennules"="Antennules","Temp C Low"="Temp C Low","BP (mb) Low"="BP (mb) Low","BP (mb) High"="BP (mb) High",
                                   "Wind Low (Km/hr)"="Wind Low (Km/hr)","Wind High (Km/hr)"="Wind High (Km/hr)","Temp C High"="Temp C High","Percipitation (mm)"="Percipitation (mm)",
                                   "Humidity High"="Humidity High","Humidity Low"="Humidity Low","Body Length 1 mm"="Body Length 1 mm","Body Length 2 mm"="Body Length 2 mm",
                                   "Body Length 3 mm"="Body Length 3 mm","Fore Wing Length01mm"="Fore Wing Length01mm",
                                   "Fore Wing Length02mm"="Fore Wing Length02mm","Fore Wing Width01mm"="Fore Wing Width01mm","Fore Wing Width02mm"="Fore Wing Width02mm",
                                   "Fore Wing Width03mm"="Fore Wing Width03mm","Fore Wing Width04mm"="Fore Wing Width04mm","Fore Wing Width05mm"="Fore Wing Width05mm",
                                   "Fore Wing Width06mm"="Fore Wing Width06mm","Fore Wing Width07mm"="Fore Wing Width07mm","Hind Wing Length01mm"="Hind Wing Length01mm",
                                   "HWL02mm"="HWL02mm","Hind Wing Width01mm"="Hind Wing Width01mm","Hind Wing Width02mm"="Hind Wing Width02mm","Hind Wing Width03mm"="Hind Wing Width03mm",
                                   "Hind Wing Width04mm"="Hind Wing Width04mm","Hind Wing Width05mm"="Hind Wing Width05mm","Hind Wing Width06mm"="Hind Wing Width06mm",
                                   "Hind Wing Width07mm"="Hind Wing Width07mm","Hind Wing perimeter"="Hind Wing perimeter","Hind Wing perimeter"="Hind Wing perimeter",
                                   "Body Width01mm"="Body Width01mm","Body Width02mm"="Body Width02mm","Body Width03mm"="Body Width03mm",
                                   "Telson L"="Telson L","Telson W"="Telson W","Orbital"="Orbital","W. B/t orbitals"="W. B/t orbitals","Chela L"="Chela L","Chela W"="Chela W",
                                   "chela + immovable finger"="chela + immovable finger","movable finger"="movable finger",
                                   "Librigena Length"="Librigena Length","Head thickness"="Head thickness","Body Thickness 1"="Body Thickness 1","Body Thickness 2"="Body Thickness 2",
                                   "Skull Length"="Skull Length","Skull Hieght"="Skull Hieght","Skull Width"="Skull Width","Neck"="Neck","Rib cage Length"="Rib cage Length",
                                   "Femur"="Femur","Tibia"="Tibia","Foot Length"="Foot Length","Foot Width"="Foot Width","Humerous"="Humerous","Lower arm"="Lower arm",
                                   "Digits 1"="Digits 1","Digits 2"="Digits 2","Digits 3"="Digits 3","Digits 4"="Digits 4","Tail"="Tail")),
        zaxis =list(title = switch(input$TZ,"Wing Span"="Wing Span","Total Body Length"="Total Body Length","Fore Wing Area"="Fore Wing Area",
                                   "Hind Wing Area"="Hind Wing Area","Antenna"="Antenna","Antennules"="Antennules","Temp C Low"="Temp C Low","BP (mb) Low"="BP (mb) Low","BP (mb) High"="BP (mb) High",
                                   "Wind Low (Km/hr)"="Wind Low (Km/hr)","Wind High (Km/hr)"="Wind High (Km/hr)","Temp C High"="Temp C High","Percipitation (mm)"="Percipitation (mm)",
                                   "Humidity High"="Humidity High","Humidity Low"="Humidity Low","Body Length 1 mm"="Body Length 1 mm","Body Length 2 mm"="Body Length 2 mm",
                                   "Body Length 3 mm"="Body Length 3 mm","Fore Wing Length01mm"="Fore Wing Length01mm",
                                   "Fore Wing Length02mm"="Fore Wing Length02mm","Fore Wing Width01mm"="Fore Wing Width01mm","Fore Wing Width02mm"="Fore Wing Width02mm",
                                   "Fore Wing Width03mm"="Fore Wing Width03mm","Fore Wing Width04mm"="Fore Wing Width04mm","Fore Wing Width05mm"="Fore Wing Width05mm",
                                   "Fore Wing Width06mm"="Fore Wing Width06mm","Fore Wing Width07mm"="Fore Wing Width07mm","Hind Wing Length01mm"="Hind Wing Length01mm",
                                   "HWL02mm"="HWL02mm","Hind Wing Width01mm"="Hind Wing Width01mm","Hind Wing Width02mm"="Hind Wing Width02mm","Hind Wing Width03mm"="Hind Wing Width03mm",
                                   "Hind Wing Width04mm"="Hind Wing Width04mm","Hind Wing Width05mm"="Hind Wing Width05mm","Hind Wing Width06mm"="Hind Wing Width06mm",
                                   "Hind Wing Width07mm"="Hind Wing Width07mm","Hind Wing perimeter"="Hind Wing perimeter","Hind Wing perimeter"="Hind Wing perimeter",
                                   "Body Width01mm"="Body Width01mm","Body Width02mm"="Body Width02mm","Body Width03mm"="Body Width03mm",
                                   "Telson L"="Telson L","Telson W"="Telson W","Orbital"="Orbital","W. B/t orbitals"="W. B/t orbitals","Chela L"="Chela L","Chela W"="Chela W",
                                   "chela + immovable finger"="chela + immovable finger","movable finger"="movable finger",
                                   "Librigena Length"="Librigena Length","Head thickness"="Head thickness","Body Thickness 1"="Body Thickness 1","Body Thickness 2"="Body Thickness 2",
                                   "Skull Length"="Skull Length","Skull Hieght"="Skull Hieght","Skull Width"="Skull Width","Neck"="Neck","Rib cage Length"="Rib cage Length",
                                   "Femur"="Femur","Tibia"="Tibia","Foot Length"="Foot Length","Foot Width"="Foot Width","Humerous"="Humerous","Lower arm"="Lower arm",
                                   "Digits 1"="Digits 1","Digits 2"="Digits 2","Digits 3"="Digits 3","Digits 4"="Digits 4","Tail"="Tail"))))
  })
  
  
  ##print histogram
  output$Plot1 <- downloadHandler(
    filename = function(){paste("plot1.pdf",sep=".") },
    content = function(file) {
      pdf(file)
      print(hist(switch(input$X,"Wing Span"=pointsInBounds()$`Wing Span`,"Total Body Length"=pointsInBounds()$`Total Body Length`,"Fore Wing Area"=pointsInBounds()$`Fore Wing Area`,"Hind Wing Area"=pointsInBounds()$`Hind Wing Area`,"Antenna"=pointsInBounds()$Antenna,
                        "BP (mb) Low"=pointsInBounds()$`BP (mb) Low`,"BP (mb) High"=pointsInBounds()$`BP (mb) High`,"Wind Low (Km/hr)"=pointsInBounds()$`Wind Low (Km/hr)`,"Wind High (Km/hr)"=pointsInBounds()$`Wind High (Km/hr)`,"Temp C Low"=pointsInBounds()$`Temp C Low`,"Temp C High"=pointsInBounds()$`Temp C High`,
                        "Percipitation (mm)"=pointsInBounds()$`Percipitation (mm)`,"Humidity High"=pointsInBounds()$`Humidity High`,"Humidity Low"=pointsInBounds()$`Humidity Low`,"Body Length 1 mm"=pointsInBounds()$`Body Length 1 mm`,"Body Length 2 mm"=pointsInBounds()$`Body Length 2 mm`,"Fore Wing Length01mm"=pointsInBounds()$`Fore Wing Length01mm`,
                        "Fore Wing Length02mm"=pointsInBounds()$`Fore Wing Length02mm`,"Fore Wing Width01mm"=pointsInBounds()$`Fore Wing Width01mm`,"Fore Wing Width02mm"=pointsInBounds()$`Fore Wing Width02mm`,"Fore Wing Width03mm"=pointsInBounds()$`Fore Wing Width03mm`,"Fore Wing Width04mm"=pointsInBounds()$`Fore Wing Width04mm`,"Fore Wing Width05mm"=pointsInBounds()$`Fore Wing Width05mm`,"Fore Wing Width06mm"=pointsInBounds()$`Fore Wing Width06mm`,"Fore Wing Width07mm"=pointsInBounds()$`Fore Wing Width07mm`,
                        "Hind Wing Length01mm"=pointsInBounds()$`Hind Wing Length01mm`,"HWL02mm"=pointsInBounds()$`HWL02mm`,"Hind Wing Width01mm"=pointsInBounds()$`Hind Wing Width01mm`,"Hind Wing Width02mm"=pointsInBounds()$`Hind Wing Width02mm`,"Hind Wing Width03mm"=pointsInBounds()$`Hind Wing Width03mm`,"Hind Wing Width04mm"=pointsInBounds()$`Hind Wing Width04mm`,"Hind Wing Width05mm"=pointsInBounds()$`Hind Wing Width05mm`,"Hind Wing Width06mm"=pointsInBounds()$`Hind Wing Width06mm`,
                        "Hind Wing Width07mm"=pointsInBounds()$`Hind Wing Width07mm`,"Fore Wing perimeter"=pointsInBounds()$`Fore Wing perimeter`,"Hind Wing perimeter"=pointsInBounds()$`Hind Wing perimeter`),
                 breaks = hist(plot = TRUE,switch(input$X,"Wing Span"=pointsInBounds()$`Wing Span`,"Total Body Length"=pointsInBounds()$`Total Body Length`,"Fore Wing Area"=pointsInBounds()$`Fore Wing Area`,"Hind Wing Area"=pointsInBounds()$`Hind Wing Area`,"Antenna"=pointsInBounds()$Antenna,
                                                  "BP (mb) Low"=pointsInBounds()$`BP (mb) Low`,"BP (mb) High"=pointsInBounds()$`BP (mb) High`,"Wind Low (Km/hr)"=pointsInBounds()$`Wind Low (Km/hr)`,"Wind High (Km/hr)"=pointsInBounds()$`Wind High (Km/hr)`,"Temp C Low"=pointsInBounds()$`Temp C Low`,"Temp C High"=pointsInBounds()$`Temp C High`,
                                                  "Percipitation (mm)"=pointsInBounds()$`Percipitation (mm)`,"Humidity High"=pointsInBounds()$`Humidity High`,"Humidity Low"=pointsInBounds()$`Humidity Low`,"Body Length 1 mm"=pointsInBounds()$`Body Length 1 mm`,"Body Length 2 mm"=pointsInBounds()$`Body Length 2 mm`,"Fore Wing Length01mm"=pointsInBounds()$`Fore Wing Length01mm`,
                                                  "Fore Wing Length02mm"=pointsInBounds()$`Fore Wing Length02mm`,"Fore Wing Width01mm"=pointsInBounds()$`Fore Wing Width01mm`,"Fore Wing Width02mm"=pointsInBounds()$`Fore Wing Width02mm`,"Fore Wing Width03mm"=pointsInBounds()$`Fore Wing Width03mm`,"Fore Wing Width04mm"=pointsInBounds()$`Fore Wing Width04mm`,"Fore Wing Width05mm"=pointsInBounds()$`Fore Wing Width05mm`,"Fore Wing Width06mm"=pointsInBounds()$`Fore Wing Width06mm`,"Fore Wing Width07mm"=pointsInBounds()$`Fore Wing Width07mm`,
                                                  "Hind Wing Length01mm"=pointsInBounds()$`Hind Wing Length01mm`,"HWL02mm"=pointsInBounds()$`HWL02mm`,"Hind Wing Width01mm"=pointsInBounds()$`Hind Wing Width01mm`,"Hind Wing Width02mm"=pointsInBounds()$`Hind Wing Width02mm`,"Hind Wing Width03mm"=pointsInBounds()$`Hind Wing Width03mm`,"Hind Wing Width04mm"=pointsInBounds()$`Hind Wing Width04mm`,"Hind Wing Width05mm"=pointsInBounds()$`Hind Wing Width05mm`,"Hind Wing Width06mm"=pointsInBounds()$`Hind Wing Width06mm`,
                                                  "Hind Wing Width07mm"=pointsInBounds()$`Hind Wing Width07mm`,"Fore Wing perimeter"=pointsInBounds()$`Fore Wing perimeter`,"Hind Wing perimeter"=pointsInBounds()$`Hind Wing perimeter`), 
                               breaks = 20)$breaks,
                 xlab = switch(input$X,"Wing Span"="Wing Span","Total Body Length"="Total Body Length","Fore Wing Area"="Fore Wing Area","Hind Wing Area"="Hind Wing Area","Antenna"="Antenna","Temp C Low"="Temp C Low","BP (mb) Low"="BP (mb) Low","BP (mb) High"="BP (mb) High",
                               "Wind Low (Km/hr)"="Wind Low (Km/hr)","Wind High (Km/hr)"="Wind High (Km/hr)","Temp C High"="Temp C High","Percipitation (mm)"="Percipitation (mm)","Humidity High"="Humidity High","Humidity Low"="Humidity Low","Body Length 1 mm"="Body Length 1 mm","Body Length 2 mm"="Body Length 2 mm","Fore Wing Length01mm"="Fore Wing Length01mm",
                               "Fore Wing Length02mm"="Fore Wing Length02mm","Fore Wing Width01mm"="Fore Wing Width01mm","Fore Wing Width02mm"="Fore Wing Width02mm","Fore Wing Width03mm"="Fore Wing Width03mm","Fore Wing Width04mm"="Fore Wing Width04mm","Fore Wing Width05mm"="Fore Wing Width05mm","Fore Wing Width06mm"="Fore Wing Width06mm","Fore Wing Width07mm"="Fore Wing Width07mm","Hind Wing Length01mm"="Hind Wing Length01mm","HWL02mm"="HWL02mm","Hind Wing Width01mm"="Hind Wing Width01mm","Hind Wing Width02mm"="Hind Wing Width02mm","Hind Wing Width03mm"="Hind Wing Width03mm",
                               "Fore Wing Width04mm"="Fore Wing Width04mm","Fore Wing Width05mm"="Fore Wing Width05mm","Fore Wing Width06mm"="Fore Wing Width06mm","Hind Wing Width07mm"="Hind Wing Width07mm","Fore Wing perimeter"="Fore Wing perimeter","Hind Wing perimeter"="Hind Wing perimeter"),
                 main = "",
                 ylab = "Amount of Organisms in this Area",
                 col = rainbow(30),
                 border = 'white'))
      dev.off()
    })
  ##print Density
  output$Plot2 <- downloadHandler(
    filename = function(){paste("plot1.pdf",sep=".") },
    content = function(file) {
      pdf(file)
      print(plot(density(na.omit(switch(input$DX,"Wing Span"=pointsInBounds()$`Wing Span`,"Total Body Length"=pointsInBounds()$`Total Body Length`,"Fore Wing Area"=pointsInBounds()$`Fore Wing Area`,"Hind Wing Area"=pointsInBounds()$`Hind Wing Area`,"Antenna"=pointsInBounds()$Antenna,
                                        "BP (mb) Low"=pointsInBounds()$`BP (mb) Low`,"BP (mb) High"=pointsInBounds()$`BP (mb) High`,"Wind Low (Km/hr)"=pointsInBounds()$`Wind Low (Km/hr)`,"Wind High (Km/hr)"=pointsInBounds()$`Wind High (Km/hr)`,"Temp C Low"=pointsInBounds()$`Temp C Low`,"Temp C High"=pointsInBounds()$`Temp C High`,
                                        "Percipitation (mm)"=pointsInBounds()$`Percipitation (mm)`,"Humidity High"=pointsInBounds()$`Humidity High`,"Humidity Low"=pointsInBounds()$`Humidity Low`,"Body Length 1 mm"=pointsInBounds()$`Body Length 1 mm`,"Body Length 2 mm"=pointsInBounds()$`Body Length 2 mm`,"Fore Wing Length01mm"=pointsInBounds()$`Fore Wing Length01mm`,
                                        "Fore Wing Length02mm"=pointsInBounds()$`Fore Wing Length02mm`,"Fore Wing Width01mm"=pointsInBounds()$`Fore Wing Width01mm`,"Fore Wing Width02mm"=pointsInBounds()$`Fore Wing Width02mm`,"Fore Wing Width03mm"=pointsInBounds()$`Fore Wing Width03mm`,"Fore Wing Width04mm"=pointsInBounds()$`Fore Wing Width04mm`,"Fore Wing Width05mm"=pointsInBounds()$`Fore Wing Width05mm`,"Fore Wing Width06mm"=pointsInBounds()$`Fore Wing Width06mm`,"Fore Wing Width07mm"=pointsInBounds()$`Fore Wing Width07mm`,
                                        "Hind Wing Length01mm"=pointsInBounds()$`Hind Wing Length01mm`,"HWL02mm"=pointsInBounds()$`HWL02mm`,"Hind Wing Width01mm"=pointsInBounds()$`Hind Wing Width01mm`,"Hind Wing Width02mm"=pointsInBounds()$`Hind Wing Width02mm`,"Hind Wing Width03mm"=pointsInBounds()$`Hind Wing Width03mm`,"Hind Wing Width04mm"=pointsInBounds()$`Hind Wing Width04mm`,"Hind Wing Width05mm"=pointsInBounds()$`Hind Wing Width05mm`,"Hind Wing Width06mm"=pointsInBounds()$`Hind Wing Width06mm`,
                                        "Hind Wing Width07mm"=pointsInBounds()$`Hind Wing Width07mm`,"Fore Wing perimeter"=pointsInBounds()$`Fore Wing perimeter`,"Hind Wing perimeter"=pointsInBounds()$`Hind Wing perimeter`))),
                 main = "",
                 col = rainbow(30),
                 border = 'white'))
      dev.off()
    })
  ##Tables    
  ###################################################
  ##All
  observe({
    
    Family <- if (is.null(input$Order)) character(0) else {
      filter(Data, Order %in% input$Order) %>%
        `$`('Family') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$Family[input$Family %in% Family])
    updateSelectInput(session, "Family", choices = Family,
                      selected = stillSelected)
  })
  observe({
    Species <- if (is.null(input$Order)) character(0) else {
      Data %>%
        filter(Order %in% input$Order,
               is.null(input$Family) | Family %in% input$Family) %>%
        `$`('Species') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$Species[input$Species %in% Species])
    updateSelectInput(session, "Species", choices = Species,
                      selected = stillSelected)
  })
  ##Flight
  observe({
    
    FFamily <- if (is.null(input$FOrder)) character(0) else {
      filter(Flight.Animals, Order %in% input$FOrder) %>%
        `$`('Family') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$FFamily[input$FFamily %in% Family])
    updateSelectInput(session, "FFamily", choices = FFamily,
                      selected = stillSelected)
  })
  observe({
    FSpecies <- if (is.null(input$FOrder)) character(0) else {
      Flight.Animals %>%
        filter(Order %in% input$FOrder,
               is.null(input$FFamily) | Family %in% input$FFamily) %>%
        `$`('Species') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$FSpecies[input$FSpecies %in% Species])
    updateSelectInput(session, "FSpecies", choices = FSpecies,
                      selected = stillSelected)
  })
  ##Land
  observe({
    
    LFamily <- if (is.null(input$LOrder)) character(0) else {
      filter(Land.Animals, Order %in% input$LOrder) %>%
        `$`('Family') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$LFamily[input$LFamily %in% Family])
    updateSelectInput(session, "LFamily", choices = LFamily,
                      selected = stillSelected)
  })
  observe({
    LSpecies <- if (is.null(input$LOrder)) character(0) else {
      Land.Animals %>%
        filter(Order %in% input$LOrder,
               is.null(input$LFamily) | Family %in% input$LFamily) %>%
        `$`('Species') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$LSpecies[input$LSpecies %in% Species])
    updateSelectInput(session, "LSpecies", choices = LSpecies,
                      selected = stillSelected)
  })
  ##Water
  observe({
    
    WFamily <- if (is.null(input$WOrder)) character(0) else {
      filter(Water.Animals, Order %in% input$WOrder) %>%
        `$`('Family') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$WFamily[input$WFamily %in% Family])
    updateSelectInput(session, "WFamily", choices = WFamily,
                      selected = stillSelected)
  })
  observe({
    WSpecies <- if (is.null(input$WOrder)) character(0) else {
      Water.Animals %>%
        filter(Order %in% input$WOrder,
               is.null(input$WFamily) | Family %in% input$WFamily) %>%
        `$`('Species') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$WSpecies[input$WSpecies %in% Species])
    updateSelectInput(session, "WSpecies", choices = WSpecies,
                      selected = stillSelected)
  })
  ##Plant
  observe({
    
    PFamily <- if (is.null(input$POrder)) character(0) else {
      filter(Plants, Order %in% input$POrder) %>%
        `$`('Family') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$PFamily[input$PFamily %in% Family])
    updateSelectInput(session, "PFamily", choices = PFamily,
                      selected = stillSelected)
  })
  observe({
    PSpecies <- if (is.null(input$POrder)) character(0) else {
      Plants %>%
        filter(Order %in% input$POrder,
               is.null(input$PFamily) | Family %in% input$PFamily) %>%
        `$`('Species') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$PSpecies[input$PSpecies %in% Species])
    updateSelectInput(session, "PSpecies", choices = PSpecies,
                      selected = stillSelected)
  })
  ##Micro
  observe({
    
    MFamily <- if (is.null(input$MOrder)) character(0) else {
      filter(Micro, Order %in% input$MOrder) %>%
        `$`('Family') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$MFamily[input$MFamily %in% Family])
    updateSelectInput(session, "MFamily", choices = MFamily,
                      selected = stillSelected)
  })
  observe({
    MSpecies <- if (is.null(input$MOrder)) character(0) else {
      Micro %>%
        filter(Order %in% input$MOrder,
               is.null(input$MFamily) | Family %in% input$MFamily) %>%
        `$`('Species') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$MSpecies[input$MSpecies %in% Species])
    updateSelectInput(session, "MSpecies", choices = MSpecies,
                      selected = stillSelected)
  })
  ##Extreme
  observe({
    
    EFamily <- if (is.null(input$EOrder)) character(0) else {
      filter(Extremophiles, Order %in% input$EOrder) %>%
        `$`('Family') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$EFamily[input$EFamily %in% Family])
    updateSelectInput(session, "EFamily", choices = EFamily,
                      selected = stillSelected)
  })
  observe({
    ESpecies <- if (is.null(input$EOrder)) character(0) else {
      Extremophiles %>%
        filter(Order %in% input$EOrder,
               is.null(input$EFamily) | Family %in% input$EFamily) %>%
        `$`('Species') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$ESpecies[input$ESpecies %in% Species])
    updateSelectInput(session, "ESpecies", choices = ESpecies,
                      selected = stillSelected)
  })
  ##Non
  observe({
    
    Sub <- if (is.null(input$Cat)) character(0) else {
      filter(Non, Order %in% input$Cat) %>%
        `$`('Family') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$Sub[input$Sub %in% Family])
    updateSelectInput(session, "Sub", choices = Sub,
                      selected = stillSelected)
  })
  observe({
    Name <- if (is.null(input$Name)) character(0) else {
      Non %>%
        filter(Order %in% input$Cat,
               is.null(input$Sub) | Family %in% input$Sub) %>%
        `$`('Species') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$Name[input$Name %in% Species])
    updateSelectInput(session, "Name", choices = Name,
                      selected = stillSelected)
  })
  ##All Data
  observe({
    
    AFamily <- if (is.null(input$AOrder)) character(0) else {
      filter(Data, Order %in% input$AOrder) %>%
        `$`('Family') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$AFamily[input$AFamily %in% Family])
    updateSelectInput(session, "AFamily", choices = AFamily,
                      selected = stillSelected)
  })
  observe({
    ASpecies <- if (is.null(input$AOrder)) character(0) else {
      Data %>%
        filter(Order %in% input$AOrder,
               is.null(input$AFamily) | Family %in% input$AFamily) %>%
        `$`('Species') %>%
        unique() %>%
        sort()
    }
    stillSelected <- isolate(input$ASpecies[input$ASpecies %in% Species])
    updateSelectInput(session, "ASpecies", choices = ASpecies,
                      selected = stillSelected)
  })
  
  ##All
  output$tbl_A <- DT::renderDataTable({
    DF<-Data%>%
      filter(
        Data$`Temp C Low` >= input$temp[1],
        Data$`Temp C High` <= input$temp[2],
        Data$`Wind Low (Km/hr)` >= input$Wind[1],
        Data$`Wind High (Km/hr)` <= input$Wind[2],
        Data$`BP (mb) Low` >= input$Bp[1],
        Data$`BP (mb) High` <= input$Bp[2],
        Data$`Humidity Low`>= input$Humid[1],
        Data$`Humidity High` <= input$Humid[2],
        Data$`Percipitation (mm)`>= input$precip[1],
        Data$`Percipitation (mm)` <= input$precip[2],
        Data$`Total Body Length`<=input$TBL[2],
        Data$`Total Body Length`>=input$TBL[1],
        Data$`Wing Span`<=input$WS[2],
        Data$`Wing Span`>=input$WS[1],
        Data$`Fore Wing Area`<=input$FW[2],
        Data$`Fore Wing Area`>=input$FW[1],
        Data$`Hind Wing Area`<=input$HW[2],
        Data$`Hind Wing Area`>=input$HW[1],
        Data$`Number of Wings`>=input$NW[1],
        Data$`Number of Wings`<=input$NW[2],
        Data$`Status Number`>=input$EX[1],
        Data$`Status Number`<=input$EX[2],
        is.null(input$Order) | Order %in% input$Order,
        is.null(input$Family) | Family %in% input$Family,
        is.null(input$Species) | Species %in% input$Species,
        is.null(input$Env) | Environment %in% input$Env
      )%>%
      DT::datatable(Data, escape = FALSE,extensions = c('FixedColumns','Scroller','KeyTable','ColReorder','Buttons'), 
                    
                    options = list(
                      dom = 'Bfrtip',
                      scrollX = TRUE,
                      fixedColumns = list(leftColumns = 2),
                      pageLength = 80,
                      lengthMenu = c(10, 15, 20), 
                      deferRender = TRUE,
                      scrollY = 250,
                      scroller = TRUE,
                      keys = TRUE,
                      buttons = list('copy', 'csv', 'excel', 'pdf', 'print',list(extend = 'colvis', columns = c(2:87),collectionLayout= 'fixed four-column')),
                      colReorder = TRUE))
    
  }, server = FALSE)
  
  ##Flight
  output$tbl_F <- DT::renderDataTable({
    DF<-Flight.Animals%>%
      filter(
        Flight.Animals$`Temp C Low` >= input$Ftemp[1],
        Flight.Animals$`Temp C High` <= input$Ftemp[2],
        Flight.Animals$`Wind Low (Km/hr)` >= input$FWind[1],
        Flight.Animals$`Wind High (Km/hr)` <= input$FWind[2],
        Flight.Animals$`BP (mb) Low` >= input$FBp[1],
        Flight.Animals$`BP (mb) High` <= input$FBp[2],
        Flight.Animals$`Humidity Low`>= input$FHumid[1],
        Flight.Animals$`Humidity High` <= input$FHumid[2],
        Flight.Animals$`Percipitation (mm)`>= input$Fprecip[1],
        Flight.Animals$`Percipitation (mm)` <= input$Fprecip[2],
        Flight.Animals$`Total Body Length`<=input$FTBL[2],
        Flight.Animals$`Total Body Length`>=input$FTBL[1],
        Flight.Animals$`Wing Span`<=input$FWS[2],
        Flight.Animals$`Wing Span`>=input$FWS[1],
        Flight.Animals$`Fore Wing Area`<=input$FFW[2],
        Flight.Animals$`Fore Wing Area`>=input$FFW[1],
        Flight.Animals$`Hind Wing Area`<=input$FHW[2],
        Flight.Animals$`Hind Wing Area`>=input$FHW[1],
        Flight.Animals$`Number of Wings`>=input$FNW[1],
        Flight.Animals$`Number of Wings`<=input$FNW[2],
        is.null(input$FOrder) | Order %in% input$FOrder,
        is.null(input$FFamily) | Family %in% input$FFamily,
        is.null(input$FSpecies) | Species %in% input$FSpecies
      )%>%
      DT::datatable(Flight.Animals, escape = FALSE,extensions = c('FixedColumns','Scroller','KeyTable','ColReorder','Buttons'), 
                    
                    options = list(
                      dom = 'Bfrtip',
                      scrollX = TRUE,
                      fixedColumns = list(leftColumns = 2),
                      pageLength = 80,
                      lengthMenu = c(10, 15, 20), 
                      deferRender = TRUE,
                      scrollY = 250,
                      scroller = TRUE,
                      keys = TRUE,
                      buttons = list('copy', 'csv', 'excel', 'pdf', 'print',list(extend = 'colvis', columns = c(2:54),collectionLayout= 'fixed three-column')),
                      colReorder = TRUE))
    
  }, server = FALSE)
  
  ##Land
  output$tbl_L <- DT::renderDataTable({
    DF<-Land.Animals%>%
      filter(
        Land.Animals$`Temp C Low` >= input$temp[1],
        Land.Animals$`Temp C High` <= input$temp[2],
        Land.Animals$`Wind Low (Km/hr)` >= input$Wind[1],
        Land.Animals$`Wind High (Km/hr)` <= input$Wind[2],
        Land.Animals$`BP (mb) Low` >= input$Bp[1],
        Land.Animals$`BP (mb) High` <= input$Bp[2],
        Land.Animals$`Humidity Low`>= input$Humid[1],
        Land.Animals$`Humidity High` <= input$Humid[2],
        Land.Animals$`Percipitation (mm)`>= input$precip[1],
        Land.Animals$`Percipitation (mm)` <= input$precip[2],
        Land.Animals$`Total Body Length`<=input$TBL[2],
        Land.Animals$`Total Body Length`>=input$TBL[1],
        Land.Animals$`Wing Span`<=input$WS[2],
        Land.Animals$`Wing Span`>=input$WS[1],
        Land.Animals$`Fore Wing Area`<=input$FW[2],
        Land.Animals$`Fore Wing Area`>=input$FW[1],
        Land.Animals$`Hind Wing Area`<=input$HW[2],
        Land.Animals$`Hind Wing Area`>=input$HW[1],
        Land.Animals$`Number of Wings`>=input$NW[1],
        Land.Animals$`Number of Wings`<=input$NW[2],
        is.null(input$LOrder) | Order %in% input$LOrder,
        is.null(input$LFamily) | Family %in% input$LFamily,
        is.null(input$LSpecies) | Species %in% input$LSpecies
      )%>%
      DT::datatable(Land.Animals, escape = FALSE,extensions = c('FixedColumns','Scroller','KeyTable','ColReorder','Buttons'), 
                    
                    options = list(
                      dom = 'Bfrtip',
                      scrollX = TRUE,
                      fixedColumns = list(leftColumns = 2),
                      pageLength = 80,
                      lengthMenu = c(10, 15, 20), 
                      deferRender = TRUE,
                      scrollY = 250,
                      scroller = TRUE,
                      keys = TRUE,
                      buttons = list('copy', 'csv', 'excel', 'pdf', 'print',list(extend = 'colvis', columns = c(2:54),collectionLayout= 'fixed three-column')),
                      colReorder = TRUE))
    
  }, server = FALSE)
  
  ##Water
  output$tbl_W <- DT::renderDataTable({
    DF<-Water.Animals%>%
      filter(
        Water.Animals$`Temp C Low` >= input$temp[1],
        Water.Animals$`Temp C High` <= input$temp[2],
        Water.Animals$`Wind Low (Km/hr)` >= input$Wind[1],
        Water.Animals$`Wind High (Km/hr)` <= input$Wind[2],
        Water.Animals$`BP (mb) Low` >= input$Bp[1],
        Water.Animals$`BP (mb) High` <= input$Bp[2],
        Water.Animals$`Humidity Low`>= input$Humid[1],
        Water.Animals$`Humidity High` <= input$Humid[2],
        Water.Animals$`Percipitation (mm)`>= input$precip[1],
        Water.Animals$`Percipitation (mm)` <= input$precip[2],
        Water.Animals$`Total Body Length`<=input$TBL[2],
        Water.Animals$`Total Body Length`>=input$TBL[1],
        Water.Animals$`Wing Span`<=input$WS[2],
        Water.Animals$`Wing Span`>=input$WS[1],
        Water.Animals$`Fore Wing Area`<=input$FW[2],
        Water.Animals$`Fore Wing Area`>=input$FW[1],
        Water.Animals$`Hind Wing Area`<=input$HW[2],
        Water.Animals$`Hind Wing Area`>=input$HW[1],
        Water.Animals$`Number of Wings`>=input$NW[1],
        Water.Animals$`Number of Wings`<=input$NW[2],
        is.null(input$WOrder) | Order %in% input$WOrder,
        is.null(input$WFamily) | Family %in% input$WFamily,
        is.null(input$WSpecies) | Species %in% input$WSpecies
      )%>%
      DT::datatable(Water.Animals, escape = FALSE,extensions = c('FixedColumns','Scroller','KeyTable','ColReorder','Buttons'), 
                    
                    options = list(
                      dom = 'Bfrtip',
                      scrollX = TRUE,
                      fixedColumns = list(leftColumns = 2),
                      pageLength = 80,
                      lengthMenu = c(10, 15, 20), 
                      deferRender = TRUE,
                      scrollY = 250,
                      scroller = TRUE,
                      keys = TRUE,
                      buttons = list('copy', 'csv', 'excel', 'pdf', 'print',list(extend = 'colvis', columns = c(2:54),collectionLayout= 'fixed three-column')),
                      colReorder = TRUE))
    
  }, server = FALSE)
  
  ##Plant
  output$tbl_P <- DT::renderDataTable({
    DF<-Plants%>%
      filter(
        Plants$`Temp C Low` >= input$temp[1],
        Plants$`Temp C High` <= input$temp[2],
        Plants$`Wind Low (Km/hr)` >= input$Wind[1],
        Plants$`Wind High (Km/hr)` <= input$Wind[2],
        Plants$`BP (mb) Low` >= input$Bp[1],
        Plants$`BP (mb) High` <= input$Bp[2],
        Plants$`Humidity Low`>= input$Humid[1],
        Plants$`Humidity High` <= input$Humid[2],
        Plants$`Percipitation (mm)`>= input$precip[1],
        Plants$`Percipitation (mm)` <= input$precip[2],
        Plants$`Total Body Length`<=input$TBL[2],
        Plants$`Total Body Length`>=input$TBL[1],
        Plants$`Wing Span`<=input$WS[2],
        Plants$`Wing Span`>=input$WS[1],
        Plants$`Fore Wing Area`<=input$FW[2],
        Plants$`Fore Wing Area`>=input$FW[1],
        Plants$`Hind Wing Area`<=input$HW[2],
        Plants$`Hind Wing Area`>=input$HW[1],
        Plants$`Number of Wings`>=input$NW[1],
        Plants$`Number of Wings`<=input$NW[2],
        is.null(input$POrder) | Order %in% input$POrder,
        is.null(input$PFamily) | Family %in% input$PFamily,
        is.null(input$PSpecies) | Species %in% input$PSpecies
      )%>%
      DT::datatable(Plants, escape = FALSE,extensions = c('FixedColumns','Scroller','KeyTable','ColReorder','Buttons'), 
                    
                    options = list(
                      dom = 'Bfrtip',
                      scrollX = TRUE,
                      fixedColumns = list(leftColumns = 2),
                      pageLength = 80,
                      lengthMenu = c(10, 15, 20), 
                      deferRender = TRUE,
                      scrollY = 250,
                      scroller = TRUE,
                      keys = TRUE,
                      buttons = list('copy', 'csv', 'excel', 'pdf', 'print',list(extend = 'colvis', columns = c(2:54),collectionLayout= 'fixed three-column')),
                      colReorder = TRUE))
    
  }, server = FALSE)
  
  ##Micro
  output$tbl_M <- DT::renderDataTable({
    DF<-Micro%>%
      filter(
        Micro$`Temp C Low` >= input$temp[1],
        Micro$`Temp C High` <= input$temp[2],
        Micro$`Wind Low (Km/hr)` >= input$Wind[1],
        Micro$`Wind High (Km/hr)` <= input$Wind[2],
        Micro$`BP (mb) Low` >= input$Bp[1],
        Micro$`BP (mb) High` <= input$Bp[2],
        Micro$`Humidity Low`>= input$Humid[1],
        Micro$`Humidity High` <= input$Humid[2],
        Micro$`Percipitation (mm)`>= input$precip[1],
        Micro$`Percipitation (mm)` <= input$precip[2],
        Micro$`Total Body Length`<=input$TBL[2],
        Micro$`Total Body Length`>=input$TBL[1],
        Micro$`Wing Span`<=input$WS[2],
        Micro$`Wing Span`>=input$WS[1],
        Micro$`Fore Wing Area`<=input$FW[2],
        Micro$`Fore Wing Area`>=input$FW[1],
        Micro$`Hind Wing Area`<=input$HW[2],
        Micro$`Hind Wing Area`>=input$HW[1],
        Micro$`Number of Wings`>=input$NW[1],
        Micro$`Number of Wings`<=input$NW[2],
        is.null(input$MOrder) | Order %in% input$MOrder,
        is.null(input$MFamily) | Family %in% input$MFamily,
        is.null(input$MSpecies) | Species %in% input$MSpecies
      )%>%
      DT::datatable(Micro, escape = FALSE,extensions = c('FixedColumns','Scroller','KeyTable','ColReorder','Buttons'), 
                    
                    options = list(
                      dom = 'Bfrtip',
                      scrollX = TRUE,
                      fixedColumns = list(leftColumns = 2),
                      pageLength = 80,
                      lengthMenu = c(10, 15, 20), 
                      deferRender = TRUE,
                      scrollY = 250,
                      scroller = TRUE,
                      keys = TRUE,
                      buttons = list('copy', 'csv', 'excel', 'pdf', 'print',list(extend = 'colvis', columns = c(2:54),collectionLayout= 'fixed three-column')),
                      colReorder = TRUE))
    
  }, server = FALSE)
  
  ##Extreme
  output$tbl_E <- DT::renderDataTable({
    DF<-Extremophiles%>%
      filter(
        Extremophiles$`Temp C Low` >= input$temp[1],
        Extremophiles$`Temp C High` <= input$temp[2],
        Extremophiles$`Wind Low (Km/hr)` >= input$Wind[1],
        Extremophiles$`Wind High (Km/hr)` <= input$Wind[2],
        Extremophiles$`BP (mb) Low` >= input$Bp[1],
        Extremophiles$`BP (mb) High` <= input$Bp[2],
        Extremophiles$`Humidity Low`>= input$Humid[1],
        Extremophiles$`Humidity High` <= input$Humid[2],
        Extremophiles$`Percipitation (mm)`>= input$precip[1],
        Extremophiles$`Percipitation (mm)` <= input$precip[2],
        Extremophiles$`Total Body Length`<=input$TBL[2],
        Extremophiles$`Total Body Length`>=input$TBL[1],
        Extremophiles$`Wing Span`<=input$WS[2],
        Extremophiles$`Wing Span`>=input$WS[1],
        Extremophiles$`Fore Wing Area`<=input$FW[2],
        Extremophiles$`Fore Wing Area`>=input$FW[1],
        Extremophiles$`Hind Wing Area`<=input$HW[2],
        Extremophiles$`Hind Wing Area`>=input$HW[1],
        Extremophiles$`Number of Wings`>=input$NW[1],
        Extremophiles$`Number of Wings`<=input$NW[2],
        is.null(input$EOrder) | Order %in% input$EOrder,
        is.null(input$EFamily) | Family %in% input$EFamily,
        is.null(input$ESpecies) | Species %in% input$ESpecies
      )%>%
      DT::datatable(Extremophiles, escape = FALSE, extensions = c('FixedColumns','Scroller','KeyTable','ColReorder','Buttons'), 
                    
                    options = list(
                      dom = 'Bfrtip',
                      scrollX = TRUE,
                      fixedColumns = list(leftColumns = 2),
                      pageLength = 80,
                      lengthMenu = c(10, 15, 20), 
                      deferRender = TRUE,
                      scrollY = 250,
                      scroller = TRUE,
                      keys = TRUE,
                      buttons = list('copy', 'csv', 'excel', 'pdf', 'print',list(extend = 'colvis', columns = c(2:54),collectionLayout= 'fixed three-column')),
                      colReorder = TRUE))
    
  }, server = FALSE)
  
  ##Non
  output$tbl_N <- DT::renderDataTable({
    DF<-Non%>%
      filter(
        Non$`Temp C Low` >= input$temp[1],
        Non$`Temp C High` <= input$temp[2],
        Non$`Wind Low (Km/hr)` >= input$Wind[1],
        Non$`Wind High (Km/hr)` <= input$Wind[2],
        Non$`BP (mb) Low` >= input$Bp[1],
        Non$`BP (mb) High` <= input$Bp[2],
        Non$`Humidity Low`>= input$Humid[1],
        Non$`Humidity High` <= input$Humid[2],
        Non$`Percipitation (mm)`>= input$precip[1],
        Non$`Percipitation (mm)` <= input$precip[2],
        Non$`Total Body Length`<=input$TBL[2],
        Non$`Total Body Length`>=input$TBL[1],
        Non$`Wing Span`<=input$WS[2],
        Non$`Wing Span`>=input$WS[1],
        Non$`Fore Wing Area`<=input$FW[2],
        Non$`Fore Wing Area`>=input$FW[1],
        Non$`Hind Wing Area`<=input$HW[2],
        Non$`Hind Wing Area`>=input$HW[1],
        Non$`Number of Wings`>=input$NW[1],
        Non$`Number of Wings`<=input$NW[2],
        is.null(input$Cat) | Order %in% input$Cat,
        is.null(input$Sub) | Family %in% input$Sub,
        is.null(input$Name) | Species %in% input$Name
      )%>%
      DT::datatable(Non, escape = FALSE,extensions = c('FixedColumns','Scroller','KeyTable','ColReorder','Buttons'), 
                    
                    options = list(
                      dom = 'Bfrtip',
                      scrollX = TRUE,
                      fixedColumns = list(leftColumns = 2),
                      pageLength = 80,
                      lengthMenu = c(10, 15, 20), 
                      deferRender = TRUE,
                      scrollY = 250,
                      scroller = TRUE,
                      keys = TRUE,
                      buttons = list('copy', 'csv', 'excel', 'pdf', 'print',list(extend = 'colvis', columns = c(2:54),collectionLayout= 'fixed three-column')),
                      colReorder = TRUE))
    
  }, server = FALSE)
  
  output$tbl_Test <- DT::renderDataTable({
    DF<-pointsInBounds()
    DT::datatable(pointsInBounds(), escape = FALSE,
                  extensions = c('FixedColumns','Scroller','KeyTable','ColReorder','Buttons'), 
                  options = list(
                    dom = 'Bfrtip',
                    scrollX = TRUE,
                    #fixedColumns = list(leftColumns = 2),
                    pageLength = 80,
                    lengthMenu = c(10, 15, 20), 
                    deferRender = TRUE,
                    scrollY = 200,
                    scroller = TRUE,
                    keys = TRUE,
                    buttons = list('copy', 'csv', 'excel', 'pdf', 'print'),
                    colReorder = TRUE))
  }, server = FALSE)
  #SearchTree----
  
  output$Hierarchy <- renderUI({
    Hierarchy=names(m[,c(2:8)])
    selectizeInput("Hierarchy","Tree Hierarchy",
                   choices = Hierarchy,multiple=T,selected = Hierarchy,
                   options=list(plugins=list('drag_drop','remove_button')))
  })
  
  network <- reactiveValues()
  
  observeEvent(input$d3_update,{
    network$nodes <- unlist(input$d3_update$.nodesData)
    activeNode<-input$d3_update$.activeNode
    if(!is.null(activeNode)) network$click <- jsonlite::fromJSON(activeNode)
  })
  
  
  TreeStruct=eventReactive(network$nodes,{
    df=m
    if(is.null(network$nodes)){
      df=m
    }else{
      
      x.filter=tree.filter(network$nodes,m)
      df=ddply(x.filter,.(ID),function(a.x){m%>%filter_(.dots = list(a.x$FILTER))%>%distinct})
    }
    df
  })
  
  observeEvent(input$Hierarchy,{
    output$d3 <- renderD3tree({
      if(is.null(input$Hierarchy)){
        p=m
      }else{
        p=m%>%select(one_of(c(input$Hierarchy,"NEWCOL")))%>%unique
      }
      
      d3tree(data = list(root = df2tree(struct = p,rootname = "PeTaL"), layout = 'collapse'),activeReturn = c('name','value','depth','id'),height = 18)
    })
  })
  
  
  output$table <- DT::renderDataTable({
    DT::datatable(TreeStruct()%>%select(c(-NEWCOL,-X__1,-ID)), escape = FALSE,
                  extensions = c('FixedColumns','Scroller','KeyTable','ColReorder','Buttons'), 
                  options = list(bInfo=F,
                                 dom = 'Bfrtip',
                                 scrollX = TRUE,
                                 pageLength = 80, 
                                 deferRender = TRUE,
                                 scrollY = 360,
                                 scroller = TRUE,
                                 keys = TRUE,
                                 buttons = list('copy', 'csv', 'excel', 'pdf', 'print'),
                                 colReorder = TRUE))
  }, server = TRUE)
  
  output$progressBox2 <- renderInfoBox({
    infoBox(
      "Specimens in search", value=nrow(TreeStruct()), icon = icon("list"),
      color = if (nrow(TreeStruct()) >= 100) "red" else if(nrow(TreeStruct()) >= 50) "yellow" else "green"
    )
  })
  
  output$TreeMap<-renderHighchart({TM1<-hctreemap2(data = M,
                                                   group_vars = c("Status","Class","Order", "Family","Species"),
                                                   size_var = "n",
                                                   color_var = "n",
                                                   layoutAlgorithm = "squarified",
                                                   levelIsConstant = T,
                                                   levels = list(
                                                     list(level = 1, dataLabels = list(enabled = T)),
                                                     list(level = 2, dataLabels = list(enabled = F)),
                                                     list(level = 3, dataLabels = list(enabled = F)),
                                                     list(level = 4, dataLabels = list(enabled = F)),
                                                     list(level = 5, dataLabels = list(enabled = F))
                                                   )) %>% 
    hc_colorAxis(minColor = brewer.pal(9, "Blues")[5],
                 maxColor = brewer.pal(9, "Greens")[8]) %>% 
    hc_tooltip(pointFormat = "<b>{point.name}</b>:<br>
               Number Measured: {point.value:,.0f}")
  })
  })