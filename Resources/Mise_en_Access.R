library(RODBC)
library(data.table)
library(stringr)

list_fichiers <- list.files(path = './',pattern = '.csv')
names(list_fichiers) <-  list_fichiers
list_read <- list()
list_read <- lapply(list_fichiers, function(x){
  a <- fread(x)
  a
})

noms <- str_remove_all(list_fichiers,'.csv')
names(noms) <- list_fichiers

for (x in list_fichiers) {
  assign(noms[[x]], list_read[[x]])
}

con <- odbcConnectAccess2007(list.files(path = './', pattern = '.accdb')[1])  

sqlSave(con, A_1_Arrêts_Génériques)
sqlSave(con, A_2_Arrêts_Physiques)
sqlSave(con, B_1_Lignes)
sqlSave(con, B_2_Sous_Lignes)
sqlSave(con, C_1_Itinéraire)
sqlSave(con, C_2_Itinéraire_Arc)
sqlSave(con, C_3_Courses)
sqlSave(con, D_1_Service_Dates)
sqlSave(con, D_2_Service_Jourtype)
sqlSave(con, E_1_Nombre_Passage_AG)
sqlSave(con, E_2_Nombre_Courses_Lignes)
sqlSave(con, E_3_Nombre_Courses_SousLignes)
sqlSave(con, F_1_Fréquences_Périodes_SousLignes)
sqlSave(con, G_1_GOAL_Onlet_Train)
sqlSave(con, G_2_GOAL_Onlet_TrainMarche)
sqlSave(con, G_3_Itinéraire_pour_GET)
sqlSave(con, zzz_BaseFer_1_Gares_Manquantes)
sqlSave(con, zzz_BaseFer_2_Arcs_Manquants)

odbcCloseAll()
