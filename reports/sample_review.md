# QA sample review

Deterministic stratified review sample (`seed=42`).

For each row, inspect the listed source image in the local extracted/ZIP dataset and compare it with the recorded GT. One-shot rows also list the fixed normal reference.

## Anomaly zero-shot

1. `anomaly-3e5af964721e0275` — **glass-insulator**
   - image: `glass-insulator/test/missingcap/Fotos 21-10-2020_DJI_0566_cadeia_isolador_vidro_1639.jpg`
   - GT: `"anomalous"`
2. `anomaly-7f829c141de20b15` — **glass-insulator**
   - image: `glass-insulator/test/good/Fotos 01-12-2020_DJI_0465_cadeia_isolador_vidro_74.jpg`
   - GT: `"normal"`
3. `anomaly-65a078f861a6d44c` — **glass-insulator**
   - image: `glass-insulator/test/missingcap/Fotos 21-10-2020_DJI_0628_cadeia_isolador_vidro_1545.jpg`
   - GT: `"anomalous"`
4. `anomaly-ec1324a1739a3496` — **glass-insulator**
   - image: `glass-insulator/test/good/Fotos 26-11-2020_DJI_0057_cadeia_isolador_vidro_2256.jpg`
   - GT: `"normal"`
5. `anomaly-fb0144ce5a04a9f7` — **lightning-rod-suspension**
   - image: `lightning-rod-suspension/test/rust/15-06-2021_DJI_0050_728.jpg`
   - GT: `"anomalous"`
6. `anomaly-c44871a0d60830ff` — **lightning-rod-suspension**
   - image: `lightning-rod-suspension/test/good/Fotos 24-11-2020_DJI_0562_suspensao_para_raio_613.jpg`
   - GT: `"normal"`
7. `anomaly-b9c391f8d2c6de5a` — **lightning-rod-suspension**
   - image: `lightning-rod-suspension/test/rust/01-06-2021_DJI_0451_139.jpg`
   - GT: `"anomalous"`
8. `anomaly-168e622f78f8af8b` — **lightning-rod-suspension**
   - image: `lightning-rod-suspension/test/good/Fotos 17-11-2020_DJI_0112_suspensao_para_raio_283.jpg`
   - GT: `"normal"`
9. `anomaly-4711d11db783d3f3` — **polymer-insulator-upper-shackle**
   - image: `polymer-insulator-upper-shackle/test/rust/01-06-2021_DJI_0569_163.jpg`
   - GT: `"anomalous"`
10. `anomaly-657ddb67fffe4a1d` — **polymer-insulator-upper-shackle**
   - image: `polymer-insulator-upper-shackle/test/good/Fotos 02-12-2020_DJI_0458_manilha_isolador_superior_157.jpg`
   - GT: `"normal"`
11. `anomaly-d36b73ef1463b98e` — **polymer-insulator-upper-shackle**
   - image: `polymer-insulator-upper-shackle/test/rust/Fotos 23-10-2020_DJI_0062_manilha_isolador_superior_1394.jpg`
   - GT: `"anomalous"`
12. `anomaly-55716cc775485dc8` — **polymer-insulator-upper-shackle**
   - image: `polymer-insulator-upper-shackle/test/good/Fotos 30-11-2020_DJI_0098_manilha_isolador_superior_1677.jpg`
   - GT: `"normal"`
13. `anomaly-3eb58a1be437dc07` — **vari-grip**
   - image: `vari-grip/test/nest/Fotos 17-11-2020_DJI_0289_vari_grip_478.jpg`
   - GT: `"anomalous"`
14. `anomaly-e8ec158bde64e957` — **vari-grip**
   - image: `vari-grip/test/good/Fotos 20-11-2020_DJI_0014_vari_grip_716.jpg`
   - GT: `"normal"`
15. `anomaly-bab29d15e097c07e` — **vari-grip**
   - image: `vari-grip/test/rust/Fotos 23-10-2020_DJI_0029_vari_grip_745.jpg`
   - GT: `"anomalous"`
16. `anomaly-d491dab72390be4c` — **vari-grip**
   - image: `vari-grip/test/good/Fotos 30-11-2020_DJI_0502_vari_grip_970.jpg`
   - GT: `"normal"`
17. `anomaly-726df5ea8654b625` — **yoke-suspension**
   - image: `yoke-suspension/test/rust/01-06-2021_DJI_0386_117.jpg`
   - GT: `"anomalous"`
18. `anomaly-c060aabd4ec8289f` — **yoke-suspension**
   - image: `yoke-suspension/test/good/Fotos 22-10-2020_DJI_0751_amarra_balancim_4641.jpg`
   - GT: `"normal"`
19. `anomaly-dafc9942c9a1bbfb` — **yoke-suspension**
   - image: `yoke-suspension/test/rust/02-06-2021_DJI_0091_202.jpg`
   - GT: `"anomalous"`
20. `anomaly-f6009a3e36dfe11a` — **yoke-suspension**
   - image: `yoke-suspension/test/good/Fotos 22-10-2020_DJI_0727_amarra_balancim_4615.jpg`
   - GT: `"normal"`

## Anomaly one-shot

1. `anomaly-3e5af964721e0275` — **glass-insulator**
   - image: `glass-insulator/test/missingcap/Fotos 21-10-2020_DJI_0566_cadeia_isolador_vidro_1639.jpg`
   - reference: `glass-insulator/train/good/Fotos 30-11-2020_DJI_0407_cadeia_isolador_vidro_2742.jpg`
   - GT: `"anomalous"`
2. `anomaly-7f829c141de20b15` — **glass-insulator**
   - image: `glass-insulator/test/good/Fotos 01-12-2020_DJI_0465_cadeia_isolador_vidro_74.jpg`
   - reference: `glass-insulator/train/good/Fotos 30-11-2020_DJI_0407_cadeia_isolador_vidro_2742.jpg`
   - GT: `"normal"`
3. `anomaly-65a078f861a6d44c` — **glass-insulator**
   - image: `glass-insulator/test/missingcap/Fotos 21-10-2020_DJI_0628_cadeia_isolador_vidro_1545.jpg`
   - reference: `glass-insulator/train/good/Fotos 30-11-2020_DJI_0407_cadeia_isolador_vidro_2742.jpg`
   - GT: `"anomalous"`
4. `anomaly-ec1324a1739a3496` — **glass-insulator**
   - image: `glass-insulator/test/good/Fotos 26-11-2020_DJI_0057_cadeia_isolador_vidro_2256.jpg`
   - reference: `glass-insulator/train/good/Fotos 30-11-2020_DJI_0407_cadeia_isolador_vidro_2742.jpg`
   - GT: `"normal"`
5. `anomaly-fb0144ce5a04a9f7` — **lightning-rod-suspension**
   - image: `lightning-rod-suspension/test/rust/15-06-2021_DJI_0050_728.jpg`
   - reference: `lightning-rod-suspension/train/good/Fotos 03-12-2020_DJI_0576_suspensao_para_raio_57.jpg`
   - GT: `"anomalous"`
6. `anomaly-c44871a0d60830ff` — **lightning-rod-suspension**
   - image: `lightning-rod-suspension/test/good/Fotos 24-11-2020_DJI_0562_suspensao_para_raio_613.jpg`
   - reference: `lightning-rod-suspension/train/good/Fotos 03-12-2020_DJI_0576_suspensao_para_raio_57.jpg`
   - GT: `"normal"`
7. `anomaly-b9c391f8d2c6de5a` — **lightning-rod-suspension**
   - image: `lightning-rod-suspension/test/rust/01-06-2021_DJI_0451_139.jpg`
   - reference: `lightning-rod-suspension/train/good/Fotos 03-12-2020_DJI_0576_suspensao_para_raio_57.jpg`
   - GT: `"anomalous"`
8. `anomaly-168e622f78f8af8b` — **lightning-rod-suspension**
   - image: `lightning-rod-suspension/test/good/Fotos 17-11-2020_DJI_0112_suspensao_para_raio_283.jpg`
   - reference: `lightning-rod-suspension/train/good/Fotos 03-12-2020_DJI_0576_suspensao_para_raio_57.jpg`
   - GT: `"normal"`
9. `anomaly-4711d11db783d3f3` — **polymer-insulator-upper-shackle**
   - image: `polymer-insulator-upper-shackle/test/rust/01-06-2021_DJI_0569_163.jpg`
   - reference: `polymer-insulator-upper-shackle/train/good/Fotos 20-11-2020_DJI_0138_manilha_isolador_superior_1316.jpg`
   - GT: `"anomalous"`
10. `anomaly-657ddb67fffe4a1d` — **polymer-insulator-upper-shackle**
   - image: `polymer-insulator-upper-shackle/test/good/Fotos 02-12-2020_DJI_0458_manilha_isolador_superior_157.jpg`
   - reference: `polymer-insulator-upper-shackle/train/good/Fotos 20-11-2020_DJI_0138_manilha_isolador_superior_1316.jpg`
   - GT: `"normal"`
11. `anomaly-d36b73ef1463b98e` — **polymer-insulator-upper-shackle**
   - image: `polymer-insulator-upper-shackle/test/rust/Fotos 23-10-2020_DJI_0062_manilha_isolador_superior_1394.jpg`
   - reference: `polymer-insulator-upper-shackle/train/good/Fotos 20-11-2020_DJI_0138_manilha_isolador_superior_1316.jpg`
   - GT: `"anomalous"`
12. `anomaly-55716cc775485dc8` — **polymer-insulator-upper-shackle**
   - image: `polymer-insulator-upper-shackle/test/good/Fotos 30-11-2020_DJI_0098_manilha_isolador_superior_1677.jpg`
   - reference: `polymer-insulator-upper-shackle/train/good/Fotos 20-11-2020_DJI_0138_manilha_isolador_superior_1316.jpg`
   - GT: `"normal"`
13. `anomaly-3eb58a1be437dc07` — **vari-grip**
   - image: `vari-grip/test/nest/Fotos 17-11-2020_DJI_0289_vari_grip_478.jpg`
   - reference: `vari-grip/train/good/Fotos 01-12-2020_DJI_0078_vari_grip_49.jpg`
   - GT: `"anomalous"`
14. `anomaly-e8ec158bde64e957` — **vari-grip**
   - image: `vari-grip/test/good/Fotos 20-11-2020_DJI_0014_vari_grip_716.jpg`
   - reference: `vari-grip/train/good/Fotos 01-12-2020_DJI_0078_vari_grip_49.jpg`
   - GT: `"normal"`
15. `anomaly-bab29d15e097c07e` — **vari-grip**
   - image: `vari-grip/test/rust/Fotos 23-10-2020_DJI_0029_vari_grip_745.jpg`
   - reference: `vari-grip/train/good/Fotos 01-12-2020_DJI_0078_vari_grip_49.jpg`
   - GT: `"anomalous"`
16. `anomaly-d491dab72390be4c` — **vari-grip**
   - image: `vari-grip/test/good/Fotos 30-11-2020_DJI_0502_vari_grip_970.jpg`
   - reference: `vari-grip/train/good/Fotos 01-12-2020_DJI_0078_vari_grip_49.jpg`
   - GT: `"normal"`
17. `anomaly-726df5ea8654b625` — **yoke-suspension**
   - image: `yoke-suspension/test/rust/01-06-2021_DJI_0386_117.jpg`
   - reference: `yoke-suspension/train/good/Fotos 16-12-2020_DJI_0403_amarra_balancim_2995.jpg`
   - GT: `"anomalous"`
18. `anomaly-c060aabd4ec8289f` — **yoke-suspension**
   - image: `yoke-suspension/test/good/Fotos 22-10-2020_DJI_0751_amarra_balancim_4641.jpg`
   - reference: `yoke-suspension/train/good/Fotos 16-12-2020_DJI_0403_amarra_balancim_2995.jpg`
   - GT: `"normal"`
19. `anomaly-dafc9942c9a1bbfb` — **yoke-suspension**
   - image: `yoke-suspension/test/rust/02-06-2021_DJI_0091_202.jpg`
   - reference: `yoke-suspension/train/good/Fotos 16-12-2020_DJI_0403_amarra_balancim_2995.jpg`
   - GT: `"anomalous"`
20. `anomaly-f6009a3e36dfe11a` — **yoke-suspension**
   - image: `yoke-suspension/test/good/Fotos 22-10-2020_DJI_0727_amarra_balancim_4615.jpg`
   - reference: `yoke-suspension/train/good/Fotos 16-12-2020_DJI_0403_amarra_balancim_2995.jpg`
   - GT: `"normal"`

## Structured explainability

1. `defect-9740c5bc33a7116b` — **glass-insulator**
   - image: `defect_supervised/glass-insulator/val/missing-cap/Fotos 21-10-2020_DJI_0531_cadeia_isolador_vidro_1611.jpg`
   - GT: `{"anomaly": "anomalous", "defect_class": "missing-cap"}`
2. `defect-89dd7d65eef30b53` — **glass-insulator**
   - image: `defect_supervised/glass-insulator/val/good/Fotos 25-11-2020_DJI_0219_cadeia_isolador_vidro_1880.jpg`
   - GT: `{"anomaly": "normal", "defect_class": "normal"}`
3. `defect-d2ebfb90e107dbbb` — **glass-insulator**
   - image: `defect_supervised/glass-insulator/val/missing-cap/Fotos 21-10-2020_DJI_0561_cadeia_isolador_vidro_1633.jpg`
   - GT: `{"anomaly": "anomalous", "defect_class": "missing-cap"}`
4. `defect-6ddd99e19b754507` — **glass-insulator**
   - image: `defect_supervised/glass-insulator/val/good/Fotos 16-11-2020_DJI_0018_cadeia_isolador_vidro_1497.jpg`
   - GT: `{"anomaly": "normal", "defect_class": "normal"}`
5. `defect-c36179efc7abe7a3` — **lightning-rod-suspension**
   - image: `defect_supervised/lightning-rod-suspension/val/good/Fotos 20-11-2020_DJI_0244_suspensao_para_raio_492.jpg`
   - GT: `{"anomaly": "normal", "defect_class": "normal"}`
6. `defect-4f08e37919112315` — **lightning-rod-suspension**
   - image: `defect_supervised/lightning-rod-suspension/val/rust/15-06-2021_DJI_0243_777.jpg`
   - GT: `{"anomaly": "anomalous", "defect_class": "rust"}`
7. `defect-09d097e2280abfe6` — **lightning-rod-suspension**
   - image: `defect_supervised/lightning-rod-suspension/val/good/Fotos 04-12-2020_DJI_1000 (12)_suspensao_para_raio_77.jpg`
   - GT: `{"anomaly": "normal", "defect_class": "normal"}`
8. `defect-48b240878c1cec87` — **lightning-rod-suspension**
   - image: `defect_supervised/lightning-rod-suspension/val/rust/08-06-2021_DJI_0270_377.jpg`
   - GT: `{"anomaly": "anomalous", "defect_class": "rust"}`
9. `defect-ab3d52a3de614c46` — **polymer-insulator-upper-shackle**
   - image: `defect_supervised/polymer-insulator-upper-shackle/val/good/Fotos 20-10-2020_DJI_0226_manilha_isolador_superior_1293.jpg`
   - GT: `{"anomaly": "normal", "defect_class": "normal"}`
10. `defect-4d23d643907b404a` — **polymer-insulator-upper-shackle**
   - image: `defect_supervised/polymer-insulator-upper-shackle/val/rust/01-06-2021_DJI_0036_8.jpg`
   - GT: `{"anomaly": "anomalous", "defect_class": "rust"}`
11. `defect-1f83b2bfedb81d8e` — **polymer-insulator-upper-shackle**
   - image: `defect_supervised/polymer-insulator-upper-shackle/val/good/Fotos 23-11-2020_DJI_0012_manilha_isolador_superior_1456.jpg`
   - GT: `{"anomaly": "normal", "defect_class": "normal"}`
12. `defect-4064fd1ed86ee60c` — **polymer-insulator-upper-shackle**
   - image: `defect_supervised/polymer-insulator-upper-shackle/val/rust/01-06-2021_DJI_0548_160.jpg`
   - GT: `{"anomaly": "anomalous", "defect_class": "rust"}`
13. `defect-9089b6fcdd7abd88` — **vari-grip**
   - image: `defect_supervised/vari-grip/val/bird-nest/Fotos 17-11-2020_DJI_0304_vari_grip_480.jpg`
   - GT: `{"anomaly": "anomalous", "defect_class": "bird-nest"}`
14. `defect-898e196e3c744d29` — **vari-grip**
   - image: `defect_supervised/vari-grip/val/good/Fotos 19-10-2020_DJI_0085_vari_grip_17.jpg`
   - GT: `{"anomaly": "normal", "defect_class": "normal"}`
15. `defect-3a7dbb431a7cdc9a` — **vari-grip**
   - image: `defect_supervised/vari-grip/val/rust/15-06-2021_DJI_0118_741.jpg`
   - GT: `{"anomaly": "anomalous", "defect_class": "rust"}`
16. `defect-06c2deb37ba2ab38` — **vari-grip**
   - image: `defect_supervised/vari-grip/val/bird-nest/Fotos 26-11-2020_DJI_0262_vari_grip_911.jpg`
   - GT: `{"anomaly": "anomalous", "defect_class": "bird-nest"}`
17. `defect-23585cf744a18045` — **yoke-suspension**
   - image: `defect_supervised/yoke-suspension/val/good/Fotos 03-12-2020_DJI_0513_amarra_balancim_711.jpg`
   - GT: `{"anomaly": "normal", "defect_class": "normal"}`
18. `defect-4e80542e93cd0617` — **yoke-suspension**
   - image: `defect_supervised/yoke-suspension/val/rust/15-06-2021_DJI_0217_771.jpg`
   - GT: `{"anomaly": "anomalous", "defect_class": "rust"}`
19. `defect-baa01b0a2bc348b9` — **yoke-suspension**
   - image: `defect_supervised/yoke-suspension/val/good/Fotos 10-12-2020_DJI_0017_amarra_balancim_2117.jpg`
   - GT: `{"anomaly": "normal", "defect_class": "normal"}`
20. `defect-f3001572ee6201fd` — **yoke-suspension**
   - image: `defect_supervised/yoke-suspension/val/rust/09-06-2021_DJI_0474_581.jpg`
   - GT: `{"anomaly": "anomalous", "defect_class": "rust"}`
