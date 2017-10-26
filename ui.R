header <- dashboardHeader(title = tags$a(href='http://PeTaL.nasa.gov',
                                         tags$img(src='PeTaL_Icon.png',height='30',width='37')),
                          dropdownMenu(type = "tasks",
                                       taskItem(value = 20, color = "aqua",
                                                "Cargo Aircraft Pterosaur Wing Design"
                                       ),
                                       taskItem(value = 40, color = "aqua",
                                                "Bioreactor Fuel Cell"
                                       ),
                                       taskItem(value = 60, color = "aqua",
                                                "Crustacean Chitin Submarine Hull"
                                       )
                          ),
                          dropdownMenu(type = "notifications",
                                       notificationItem(
                                         text = "Bioreactor Fuel Cell has new data",
                                         icon = icon("exclamation-triangle"),
                                         status = "warning"
                                       )
                          ),
                          dropdownMenu(
                            type = "notifications", 
                            icon = icon("question-circle"),
                            badgeStatus = NULL,
                            headerText = "Help",
                            notificationItem("PeTaL User Guide (Link to Home/Guide)", icon = icon("file"),
                                             href = "http://www.google.com/"),
                            notificationItem("Contact Us (Email Link)", icon = icon("file"),
                                             href = "http://www.google.com/")
                          ))
dropdown<- dropdownMenu(type = "notifications",
                        notificationItem(
                          text = "5 new users today",
                          icon("users")
                        ),
                        notificationItem(
                          text = "12 items delivered",
                          icon("truck"),
                          status = "success"
                        ),
                        notificationItem(
                          text = "Server load at 86%",
                          icon = icon("exclamation-triangle"),
                          status = "warning"
                        )
)

sidebar <- dashboardSidebar(  
  sidebarMenu(
    id = "tabs",
    menuItem("PeTaL", tabName = "Home", icon = icon("leaf")),
    menuItem("Design Problem", tabName = "DesignProblem", icon = icon("question-circle")),
    menuItem("Interactive Map", tabName = "InteractiveMap", icon = icon("map-o")),
    #convertMenuItem(
    menuItem("Specimen Search", tabName = "SearchData", icon = icon("list")#,
             # Input directly under menuItem
             # selectInput("inputTest", "Nature Domains",
             #choices = c("All", "Air", "Land", "Water", "Plant", "Micro", "Extreme", "Inanimate"), multiple=TRUE, selectize=TRUE,
             #width = '98%')),'DataTest'
    ),
    menuItem("Data Explorer", tabName = "TreeTable", icon = icon("sitemap")),
    menuItem("Functions of Nature", tabName = "natureFunctions", icon = icon("gears")),
    menuItem("Analysis Toolkit", tabName = "AnalysisTK", icon = icon("area-chart")),
    menuItem("Model Synthesis", tabName = "ModelSynth", icon = icon("pencil")),
    menuItem("About Us", tabName = "About", icon = icon("book"))
  ))

body <- dashboardBody(tabItems(
  
  # Home tab content
  tabItem(tabName = "Home",tags$style(HTML("
        .tabbable > .nav > li[class=active]    > a {background-color: rgba(255, 255, 255, 0.46);color: #4CAF50;border: 1px solid #4CAF50;}
        .nav-tabs-custom>.nav-tabs>li.active {border-top-color: #4CAF50;}
        .nav-tabs-custom>.nav-tabs>li {border-top: 3px solid #3c8dbc;margin-bottom: -2px;margin-right: 0px;}")),
          fluidRow(width=7,background = "light-blue",column(3,align="left",tags$img(src='PeTaL_Icon.png',height='110',width='140')),column(2,align="center",h3(tags$b("Pe"),"riodic ",br(),tags$b("Ta"),"ble", br(),"of",br(),tags$b("L"),"ife")),
                   column(3,align="center",tags$a(img(src="https://www.nasa.gov/sites/all/themes/custom/nasatwo/images/nasa-logo.svg",
                                                      height="70", width="70"),href="https://www.nasa.gov/"))),
          tabsetPanel(
            tabPanel(tags$b("What is PeTaL"),
                     h4("PeTaL was made to increase biomimicry use in design. PeTaL was created by NASA through the institute", 
                        a(href="https://www.grc.nasa.gov/vibe/","VINE",icon("leaf", lib = "glyphicon")),(".")),
                     h4("The Periodic Table of Life (PeTaL) is a large public data base with the intent of increasing utilization of Bio/Nature-inspired designs.
                          PeTaL holds vast amounts of information on known natural phenomenon and organisms. PeTaL houses physical measurement,
                          functionality, and environmental data that can be used for modeling and inspiration purposes. This data offers opportunities for ",a(href="http://127.0.0.1:4084/tab-2","citizen scientists(fix link)"),(" to contribute in the comprehensive cataloguing of the natural world."))),
            tabPanel("PeTaL for Design",h4("PeTaL was created as an engineering design aid and offers users the ability to engage in the process at multiple levels including functional taxonomy, geographical/environmental, and statistical approaches towards identifying natural solutions relevant to your design goals, ")),
            tabPanel("PeTaL for Education",fluidRow(width=7,h4("PeTaL is a great tool for learning. Educators can show PeTaL to their students to get them excited about STEM. 
                                                                 A student can learn more about the natural world around them or other places around the globe to discover more about biological and
                                                                 earth sciences as well as how they apply to engineering principles."),
                                                    h3("Learn about nature near you!",br(),tags$embed(src = "Test_1.gif",hieght="50%", width="50%"),
                                                       br(),h3("See how the graphs change as you look around!",br(),tags$embed(src = "https://media.giphy.com/media/XHPR2fmsaStBC/giphy.gif",hieght="50%", width="50%"))))),
            tabPanel("PeTaL for Research",h4("PeTaL can utilize maps with built in graphing tools for environmental selection of relevant species, and includes statistical and machine learning tools in the Analysis Toolkit. 
                                               PeTaL also supports export of data and results for further investigation.")),
            tabPanel(tabName="A","Citizen Science Effort",
                     htmlOutput("frameS")))),
  
    # Design Problem tab content
  tabItem(tabName = "DesignProblem",
          h2("Select appropriate biomimetic taxonomy fields to find relevant species.")),
  
  # Interactive Map tab content
  tabItem(tabName = "InteractiveMap", 
          fluidRow(
            div(class="outer",tags$head(
              includeCSS("CustomCSS.css")),
              box(leafletOutput("map",height = "780px"),width = 12,
                  absolutePanel(id = "controls", class = "panel panel-default",
                                draggable = TRUE, top = 375, left = 15, right = "auto", bottom = "auto",
                                width = 180, height = 80,
                                checkboxInput("legend", "Show legend", FALSE),
                                checkboxInput("Table", "Show data table", FALSE)),
                  conditionalPanel("input.legend",absolutePanel(id = "controls", class = "panel panel-default",
                                                                draggable = TRUE, top = 460, left = 15, right = "auto", bottom = "auto",
                                                                width = 180, height = 200,
                                                                radioButtons("Legend","Which Legend",c("Flight Animals","Land Animals","Water Animals","Plants","Extremophiles","Micro Life","Non-Living"),
                                                                             selected = "Flight Animals",inline = F))),
                  conditionalPanel("input.Table",absolutePanel(id = "controls", class = "panel panel-default",
                                                               draggable = TRUE, top = "auto", left = 15, right = "auto", bottom = 10,
                                                               width = 800, height = 400,
                                                               br(),DT::dataTableOutput('tbl_Test'))),
                  absolutePanel(id = "controls", class = "panel panel-default",
                                draggable = TRUE, top =15, left = "auto", right = 15, bottom = "auto",
                                width = 330, height ="auto" ,
                                selectInput("point",h4(strong("Which organisms to plot")),choices = list("All",
                                                                                                         "Phylum"=c("Arthropoda",""),
                                                                                                         "Class"=c("Insecta","Reptilia","Malacostraca","Trilobita","Merostomata","Xiphosura"),
                                                                                                         "Flight Animals"=c("All Flight Animals","Anisoptera","Zygoptera","Hymenoptera","Pterosauria","Odonata",
                                                                                                                            "Diptera","Hemiptera","Saurischia","Confuciusornithiformes"),
                                                                                                         "Land Animals"=c(""),
                                                                                                         "Water Animals"=c("All Water Animals","Decapoda","Xiphosurida","Eurypterida","Eumalacostraca",
                                                                                                                           "Ptychopariida","Redlichiida","Corynexochida","Agnostida","Asaphida","Phacopida",
                                                                                                                           "Proetida"),
                                                                                                         "Plants"=c(""),
                                                                                                         "Extremophiles"=c(""),
                                                                                                         "Micro Life"=c(""),
                                                                                                         "Non-Living"=c("")))),
                  absolutePanel(id = "controls", class = "panel panel-default",
                                draggable = FALSE, top = 150, left = "auto", right = 15, bottom = "auto",
                                width = 330, height = "auto",
                                tabsetPanel(type="pills",
                                            tabPanel("Histogram",
                                                     highchartOutput("hist", height = 300),selectInput("X","X axis",c(colnames(Data[,c(18:26,33:93)])),selected = "Total Body Length"),
                                                     downloadButton("Plot1","Get Histogram")
                                            ),
                                            tabPanel("Density",
                                                     plotOutput("Den", height = 300),selectInput("DX","Density of:",c(colnames(Data[,c(18:26,33:93)])),selected = "Total Body Length"),
                                                     downloadButton("Plot2","Get Graph")),
                                            tabPanel("Scatter Plot",
                                                     rbokehOutput("Bokeh",height = 300),selectInput("BX","X axis",c(colnames(Data[,c(18:26,33:93)])),selected = "Total Body Length"),
                                                     selectInput("BY","Y axis",c(colnames(Data[,c(15:23,28:86)])),selected = "Wing Span")),
                                            tabPanel("3D Plot",tags$head(tags$style(HTML('a[data-title="Collaborate"]{display:none;}','a[data-title="Toggle show closest data on hover"]{display:none;}',
                                                                                         'a[data-title="Reset camera to last save"]{display:none;}','a[data-title="Reset camera to default"]{display:none;}',
                                                                                         'a[data-title="Zoom"]{display:none;}','a[data-title="turntable rotation"]{display:none;}','a[data-title="orbital rotation"]{display:none;}',
                                                                                         'a[data-title="Pan"]{display:none;}'))),
                                                     plotlyOutput("threeD",height = 300,width = 300),
                                                     fluidRow(column(9,
                                                                     radioButtons("pop","Hover Info",c("numbers","Info"),inline = TRUE))),
                                                     selectInput("TX","X axis",c(colnames(Data[,c(18:26,33:93)])),selected = "Total Body Length"),
                                                     selectInput("TY","Y axis",c(c(colnames(Data[,c(18:26,33:93)]))),selected = "Wing Span"),
                                                     selectInput("TZ","Z axis",c(c(colnames(Data[,c(18:26,33:93)]))),selected = "Fore Wing Area")
                                            )))
              ))
          )),
  
  # Specimen Explorer tab content
  tabItem(tabName = "SearchData",
          fluidRow(tabBox(width = 3,
                          tabPanel("Order",
                                   selectInput("AOrder", "", c("All Orders"="", structure(Data$Order, names=Data$Order)), multiple=TRUE)),
                          tabPanel("Family",selectInput("AFamily", "", c("All Families"="", structure(Data$Family, names=Data$Family)), multiple=TRUE)
                          ),
                          tabPanel("Species",
                                   selectInput("ASpecies", "", c("All Species"="", structure(Data$Species, names=Data$Species)), multiple=TRUE)
                          )),
                   box("Search by collection",width = 3,status = "primary",
                       selectInput("Collection", "", c("All Collections"="", structure(Data$Collection, names=Data$Collection)), multiple=TRUE)),
                   box("Search by Deep Time",width = 3,status = "primary",
                       selectInput("DeepTime", "", c("All DeepTime"="", structure(c("Recent","Holocene","Pleistocene","Pliocene","late Miocene","Miocene","Oligocene","Eocene","Paleocene",
                                                                                    "Late Cretaceous","middle Cretaceous","Early Cretaceous","Late Jurassic","Middle Jurassic", "Early Jurassic","Late Triassic","Middle Triassic","Early Triassic",
                                                                                    "Permian","Late Pennsylvanian","Middle Pennsylvanian","Early Pennsylvanian","Late Mississippian","Middle Mississippian","Early Mississippian",
                                                                                    "Late Devonian","Middle Devonian","Early Devonian","late Silurian","middle Silurian","early Silurian","Silurian","Late Ordovician","Middle Ordovician","Early Ordovician",
                                                                                    "late Cambrian","middle Cambrian","early Cambrian","Cambrian"))), multiple=TRUE)),
                   infoBoxOutput("progressBox",width = 3)),
          fluidRow(box(title = " ", status = "primary", width = 12, DT::dataTableOutput('tbl_1'))),
          actionButton("go", "Analyze Data")),
  # Tree Table
  tabItem(tabName = "TreeTable",
          fluidRow(box(width = 6,uiOutput("Hierarchy"), status = "primary"),infoBoxOutput("progressBox2",width = 3)),
          fluidRow(box("",width = 4,d3treeOutput(outputId="d3"), status = "primary"),box("", status = "primary", DT::dataTableOutput('table')))),
  # Functions
  tabItem(tabName = "natureFunctions",
          fluidPage(style="padding-left: 0px;padding-right: 0px;",
                    column(9,style="padding-left: 0px;padding-right: 0px;",
                           box(width = 12,height = "60px", status = "primary",h3(style="margin-top:5px;",textOutput("Name"))),
                           tabBox(width = 12,height = "600px",tabPanel("Description"),tabPanel("Environment"))),
                    column(3,style="padding-left: 0px;padding-right: 0px;",box(width = 12,height = "300px", status = "primary"),box(width = 12,height = "360px", status = "primary")),
                    fluidRow(box(width = 4,height = "300px", status = "primary"),box(width = 4,height = "300px", status = "primary"),box(width = 4,height = "300px", status = "primary"))
          )),
  # Analysis
  tabItem(tabName = "AnalysisTK",
          h2("Machine learning/statistical analysis tools."),
          fluidRow(
            box(plotOutput("cor.plot", height = 250)))),
  #Model
  tabItem(tabName = "ModelSynth",
          h2("Work bench for text editing proposals, creating flow charts/diagrams, 
             and drafting 3D models. Extract features from images and export to autocad?")),
  
  # About PeTaL
  tabItem(tabName = "About",
          tabsetPanel(
            tabPanel( "About Us", fluidRow(align="center",h4("PeTaL was created by NASA through", 
                                                             a(href="https://www.grc.nasa.gov/vibe/","VINE",icon("leaf", lib = "glyphicon")),(".")))),
            tabPanel("About Measuring"),
            tabPanel("PeTaL's Network",
                     htmlOutput("Net")),
            #https://github.com/metrumresearchgroup/d3Tree  
            tabPanel("Collection",box(width = 12,highchartOutput("TreeMap",height = "750px"))))
  )
))

dashboardPage(title=tags$head(tags$link(rel = "icon", type = "image/png", href = "PeTaL_Icon.png"),
                              tags$title("PeTaL")),
              header, sidebar, body)