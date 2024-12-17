# app.R
library(shiny)
library(shinydashboard)
library(DT)
library(randomForest)
library(xgboost)
library(glmnet)
library(dplyr)
library(pROC)

# UI Definition
ui <- dashboardPage(
  dashboardHeader(title = "Medicare Fraud Detection"),
  
  dashboardSidebar(
    sidebarMenu(
      selectInput("dataset_type", "Select Dataset Type:",
                  choices = c("DMEPOS" = "dmepos",
                              "Part B" = "part_b",
                              "Part D" = "part_d",
                              "Combined" = "combined")),
      fileInput("file", "Upload Data (RDS format)",
                accept = ".rds"),
      actionButton("analyze", "Analyze Data", 
                   class = "btn-primary",
                   style = "margin: 10px;"),
      downloadButton("download", "Download Results",
                     style = "margin: 10px;")
    )
  ),
  
  dashboardBody(
    fluidRow(
      box(
        width = 12,
        status = "info",
        solidHeader = TRUE,
        title = "Data Summary",
        tableOutput("data_summary")
      )
    ),
    
    fluidRow(
      valueBoxOutput("total_cases_box", width = 4),
      valueBoxOutput("high_risk_box", width = 4),
      valueBoxOutput("avg_prob_box", width = 4)
    ),
    
    fluidRow(
      box(
        width = 12,
        status = "primary",
        solidHeader = TRUE,
        title = "Analysis Results",
        DTOutput("results_table")
      )
    )
  )
)

# Server logic
server <- function(input, output, session) {
  # Reactive values
  rv <- reactiveValues(
    data = NULL,
    models = NULL,
    results = NULL
  )
  
  # Load data
  observeEvent(input$file, {
    req(input$file)
    tryCatch({
      rv$data <- readRDS(input$file$datapath)
      showNotification("Data loaded successfully", type = "message")
    }, error = function(e) {
      showNotification(paste("Error loading data:", e$message), type = "error")
    })
  })
  
  # Load models when dataset type changes
  observe({
    req(input$dataset_type)
    rv$models <- NULL
    
    model_path <- file.path("models", input$dataset_type)
    
    tryCatch({
      rv$models <- list(
        rf = readRDS(file.path(model_path, "rf.rds")),
        xgb = readRDS(file.path(model_path, "xgboost.rds")),
        glm = readRDS(file.path(model_path, "glmnet.rds"))
      )
      showNotification("Models loaded successfully", type = "message")
    }, error = function(e) {
      showNotification(
        paste("Error loading models:", e$message), 
        type = "error"
      )
    })
  })
  
  # Data summary
  output$data_summary <- renderTable({
    req(rv$data)
    data.frame(
      Metric = c("Number of cases", "Number of features"),
      Value = c(nrow(rv$data), ncol(rv$data))
    )
  })
  
  # Analyze data
  observeEvent(input$analyze, {
    req(rv$data, rv$models)
    
    withProgress(message = 'Analyzing data...', {
      
      # Prepare data (exclude Exclusion column if present)
      analysis_data <- rv$data[, !names(rv$data) %in% c("Exclusion")]
      
      # Get predictions from each model
      incProgress(0.3, message = "Running Random Forest model")
      rf_pred <- tryCatch({
        predict(rv$models$rf, newdata = analysis_data, type = "prob")[,"1"]
      }, error = function(e) {
        showNotification(paste("RF Error:", e$message), type = "error")
        rep(NA, nrow(analysis_data))
      })
      
      incProgress(0.3, message = "Running XGBoost model")
      xgb_pred <- tryCatch({
        predict(rv$models$xgb, as.matrix(analysis_data))
      }, error = function(e) {
        showNotification(paste("XGBoost Error:", e$message), type = "error")
        rep(NA, nrow(analysis_data))
      })
      
      incProgress(0.3, message = "Running GLMnet model")
      glm_pred <- tryCatch({
        predict(rv$models$glm, 
                newx = model.matrix(~., analysis_data)[,-1],
                type = "response")
      }, error = function(e) {
        showNotification(paste("GLMnet Error:", e$message), type = "error")
        rep(NA, nrow(analysis_data))
      })
      
      # Create results dataframe
      rv$results <- data.frame(
        Case_ID = 1:nrow(analysis_data),
        RF_Score = rf_pred,
        XGBoost_Score = xgb_pred,
        GLMnet_Score = as.vector(glm_pred)
      )
      
      # Add ensemble score
      rv$results$Ensemble_Score <- rowMeans(
        rv$results[, c("RF_Score", "XGBoost_Score", "GLMnet_Score")],
        na.rm = TRUE
      )
      
      # Add risk categories
      rv$results$Risk_Level <- cut(
        rv$results$Ensemble_Score,
        breaks = c(-Inf, 0.3, 0.7, Inf),
        labels = c("Low", "Medium", "High")
      )
      
      showNotification("Analysis completed successfully", type = "message")
    })
  })
  
  # Value boxes
  output$total_cases_box <- renderValueBox({
    req(rv$results)
    valueBox(
      nrow(rv$results),
      "Total Cases",
      icon = icon("list"),
      color = "blue"
    )
  })
  
  output$high_risk_box <- renderValueBox({
    req(rv$results)
    valueBox(
      sum(rv$results$Risk_Level == "High", na.rm = TRUE),
      "High Risk Cases",
      icon = icon("warning"),
      color = "red"
    )
  })
  
  output$avg_prob_box <- renderValueBox({
    req(rv$results)
    valueBox(
      paste0(round(mean(rv$results$Ensemble_Score, na.rm = TRUE) * 100, 1), "%"),
      "Average Risk Score",
      icon = icon("percent"),
      color = "yellow"
    )
  })
  
  # Results table
  output$results_table <- renderDT({
    req(rv$results)
    DT::datatable(
      rv$results,
      options = list(
        pageLength = 10,
        scrollX = TRUE,
        order = list(list(4, 'desc'))  # Sort by Ensemble_Score by default
      )
    ) %>%
      formatRound(
        columns = c("RF_Score", "XGBoost_Score", "GLMnet_Score", "Ensemble_Score"),
        digits = 3
      )
  })
  
  # Download handler
  output$download <- downloadHandler(
    filename = function() {
      paste0("fraud_detection_results_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(rv$results, file, row.names = FALSE)
    }
  )
}

# Run the app
shinyApp(ui = ui, server = server)