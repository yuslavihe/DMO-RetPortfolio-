document.addEventListener('DOMContentLoaded', function() {
    // Initialize Lucide icons
    lucide.createIcons();
    
    // Medical reports data
    const medicalReports = [
        {
            year: 2005,
            age: 65,
            date: "October 20, 2005",
            vitals: {
                bloodPressure: "138/88",
                bloodPressureStatus: "warning",
                heartRate: "72",
                heartRateStatus: "normal",
                respiratoryRate: "16",
                respiratoryRateStatus: "normal",
                temperature: "36.5",
                temperatureStatus: "normal",
                weight: "73.2",
                weightStatus: "normal",
                height: "172",
                bmi: "24.7",
                bmiStatus: "normal"
            },
            bloodWorkResults: {
                hemoglobin: "14.2",
                hemoglobinStatus: "normal",
                whiteBloodCells: "6.8",
                whiteBloodCellsStatus: "normal",
                platelets: "245",
                plateletsStatus: "normal",
                totalCholesterol: "205",
                totalCholesterolStatus: "warning",
                ldlCholesterol: "128",
                ldlCholesterolStatus: "warning",
                hdlCholesterol: "48",
                hdlCholesterolStatus: "normal",
                triglycerides: "142",
                triglyceridesStatus: "normal",
                fastingGlucose: "105",
                fastingGlucoseStatus: "warning",
                hba1c: "5.9",
                hba1cStatus: "warning",
                bun: "15",
                bunStatus: "normal",
                creatinine: "0.9",
                creatinineStatus: "normal",
                egfr: "86",
                egfrStatus: "normal"
            },
            imagingResults: {
                chestXray: "Normal cardiac size, clear lung fields, no abnormalities detected.",
                chestXrayStatus: "normal"
            },
            additionalNotes: "Patient reports occasional joint stiffness in knees after prolonged activity. Advised to maintain regular physical activity and monitor blood pressure at home. Recommended to limit sodium intake and increase consumption of fruits and vegetables due to borderline hypertension risk. Follow-up appointment in 6 months recommended to reassess blood pressure and glucose levels.",
            recommendedFollowUp: "Annual physical examination. Recheck blood pressure and fasting glucose in 6 months.",
            diagnosis: []
        },
        {
            year: 2007,
            age: 67,
            date: "October 17, 2007",
            vitals: {
                bloodPressure: "142/91",
                bloodPressureStatus: "abnormal",
                heartRate: "74",
                heartRateStatus: "normal",
                respiratoryRate: "17",
                respiratoryRateStatus: "normal",
                temperature: "36.7",
                temperatureStatus: "normal",
                weight: "74.5",
                weightStatus: "normal",
                height: "172",
                bmi: "25.2",
                bmiStatus: "warning"
            },
            bloodWorkResults: {
                hemoglobin: "14.1",
                hemoglobinStatus: "normal",
                whiteBloodCells: "7.0",
                whiteBloodCellsStatus: "normal",
                platelets: "238",
                plateletsStatus: "normal",
                totalCholesterol: "212",
                totalCholesterolStatus: "warning",
                ldlCholesterol: "134",
                ldlCholesterolStatus: "warning",
                hdlCholesterol: "46",
                hdlCholesterolStatus: "normal",
                triglycerides: "155",
                triglyceridesStatus: "normal",
                fastingGlucose: "110",
                fastingGlucoseStatus: "warning",
                hba1c: "6.1",
                hba1cStatus: "warning",
                bun: "16",
                bunStatus: "normal",
                creatinine: "0.92",
                creatinineStatus: "normal",
                egfr: "84",
                egfrStatus: "normal"
            },
            imagingResults: {
                chestXray: "Normal cardiac size, mild calcification of the aortic arch. Lung fields clear.",
                chestXrayStatus: "warning"
            },
            additionalNotes: "Patient reports more persistent joint pain in knees and occasional hip discomfort. Blood pressure consistently elevated on multiple readings during visit. Home blood pressure monitoring for 1 week recommended for confirmation of hypertension diagnosis. Advised to increase physical activity and reduce dietary sodium. Prediabetes indicators present - dietary counseling provided.",
            recommendedFollowUp: "Follow-up in 3 months to evaluate home blood pressure readings and reassess management strategy.",
            diagnosis: [
                {
                    name: "Essential Hypertension (Stage 1)",
                    code: "I10",
                    status: "newly diagnosed",
                    confirmatoryTests: "24-hour ambulatory blood pressure monitoring"
                }
            ],
            diagnosticTests: {
                name: "24-hour Ambulatory Blood Pressure Monitoring",
                date: "October 25, 2007",
                results: [
                    {
                        time: "Daytime Average (6:00-22:00)",
                        value: "145/92 mmHg"
                    },
                    {
                        time: "Nighttime Average (22:00-6:00)",
                        value: "138/85 mmHg"
                    },
                    {
                        time: "24-hour Average",
                        value: "143/90 mmHg"
                    },
                    {
                        time: "Blood Pressure Load",
                        value: "62% of readings > 140/90 mmHg"
                    }
                ],
                interpretation: "The 24-hour ambulatory blood pressure monitoring confirms a diagnosis of Stage 1 Hypertension. The patient demonstrates a non-dipping pattern (insufficient nighttime blood pressure reduction), which may indicate increased cardiovascular risk. Daytime, nighttime, and 24-hour averages are all above normal limits, with over 60% of readings exceeding 140/90 mmHg.",
                recommendation: "Initiation of antihypertensive therapy is recommended, beginning with a low-dose angiotensin-converting enzyme (ACE) inhibitor. Lifestyle modifications including sodium restriction (<5g daily), weight management, and regular physical activity are essential complementary interventions. Recommend re-evaluation in 4-6 weeks after treatment initiation."
            }
        },
        {
            year: 2009,
            age: 69,
            date: "October 22, 2009",
            vitals: {
                bloodPressure: "135/85",
                bloodPressureStatus: "warning",
                heartRate: "76",
                heartRateStatus: "normal",
                respiratoryRate: "17",
                respiratoryRateStatus: "normal",
                temperature: "36.6",
                temperatureStatus: "normal",
                weight: "75.1",
                weightStatus: "warning",
                height: "172",
                bmi: "25.4",
                bmiStatus: "warning"
            },
            bloodWorkResults: {
                hemoglobin: "13.9",
                hemoglobinStatus: "normal",
                whiteBloodCells: "7.2",
                whiteBloodCellsStatus: "normal",
                platelets: "232",
                plateletsStatus: "normal",
                totalCholesterol: "215",
                totalCholesterolStatus: "warning",
                ldlCholesterol: "136",
                ldlCholesterolStatus: "warning",
                hdlCholesterol: "45",
                hdlCholesterolStatus: "normal",
                triglycerides: "168",
                triglyceridesStatus: "normal",
                fastingGlucose: "115",
                fastingGlucoseStatus: "warning",
                hba1c: "6.3",
                hba1cStatus: "warning",
                bun: "17",
                bunStatus: "normal",
                creatinine: "0.94",
                creatinineStatus: "normal",
                egfr: "82",
                egfrStatus: "normal"
            },
            imagingResults: {
                chestXray: "Stable mild calcification of the aortic arch. No new findings compared to previous examination.",
                chestXrayStatus: "warning"
            },
            additionalNotes: "Hypertension now managed with low-dose ACE inhibitor (Lisinopril 10mg daily). Blood pressure is better controlled but still requires monitoring. Osteoarthritis symptoms in knees are managed with as-needed over-the-counter analgesics. Prediabetes continues - patient reports moderate success with dietary modifications but acknowledges difficulty with consistent exercise routine due to joint discomfort.",
            recommendedFollowUp: "Continue current hypertension management. Maintain 6-month follow-up schedule. Consider statin therapy for cholesterol management.",
            diagnosis: [
                {
                    name: "Essential Hypertension",
                    code: "I10",
                    status: "controlled",
                    notes: "Managed with ACE inhibitor"
                },
                {
                    name: "Prediabetes",
                    code: "R73.03",
                    status: "ongoing",
                    notes: "HbA1c 6.3%, Fasting glucose 115 mg/dL"
                }
            ]
        },
        {
            year: 2011,
            age: 71,
            date: "October 19, 2011",
            vitals: {
                bloodPressure: "138/86",
                bloodPressureStatus: "warning",
                heartRate: "78",
                heartRateStatus: "normal",
                respiratoryRate: "18",
                respiratoryRateStatus: "normal",
                temperature: "36.8",
                temperatureStatus: "normal",
                weight: "76.3",
                weightStatus: "warning",
                height: "171.5",
                bmi: "25.9",
                bmiStatus: "warning"
            },
            bloodWorkResults: {
                hemoglobin: "13.7",
                hemoglobinStatus: "normal",
                whiteBloodCells: "7.4",
                whiteBloodCellsStatus: "normal",
                platelets: "226",
                plateletsStatus: "normal",
                totalCholesterol: "195",
                totalCholesterolStatus: "normal",
                ldlCholesterol: "118",
                ldlCholesterolStatus: "warning",
                hdlCholesterol: "44",
                hdlCholesterolStatus: "normal",
                triglycerides: "174",
                triglyceridesStatus: "normal",
                fastingGlucose: "128",
                fastingGlucoseStatus: "abnormal",
                hba1c: "6.6",
                hba1cStatus: "abnormal",
                bun: "18",
                bunStatus: "normal",
                creatinine: "0.97",
                creatinineStatus: "normal",
                egfr: "79",
                egfrStatus: "normal"
            },
            imagingResults: {
                chestXray: "No significant changes from previous examination. Stable mild aortic arch calcification.",
                chestXrayStatus: "warning",
                echocardiogram: "Left ventricular ejection fraction 58%. Mild left ventricular hypertrophy consistent with history of hypertension. No significant valvular abnormalities.",
                echocardiogramStatus: "warning"
            },
            additionalNotes: "Patient reports increased daytime fatigue. Fasting glucose and HbA1c values now meet diagnostic criteria for type 2 diabetes mellitus. Currently taking Lisinopril 10mg daily and atorvastatin 10mg daily. Cholesterol values have improved with statin therapy. Echocardiogram shows early signs of hypertensive heart changes but preserved cardiac function.",
            recommendedFollowUp: "Initiate diabetes management. Schedule appointment with endocrinologist within 1 month. Continue current hypertension and cholesterol management.",
            diagnosis: [
                {
                    name: "Essential Hypertension",
                    code: "I10",
                    status: "controlled",
                    notes: "Managed with ACE inhibitor"
                },
                {
                    name: "Type 2 Diabetes Mellitus",
                    code: "E11.9",
                    status: "newly diagnosed",
                    notes: "HbA1c 6.6%, Fasting glucose 128 mg/dL",
                    confirmatoryTests: "Oral Glucose Tolerance Test (OGTT)"
                }
            ],
            diagnosticTests: {
                name: "Oral Glucose Tolerance Test (OGTT)",
                date: "October 26, 2011",
                results: [
                    {
                        time: "Fasting (0 minutes)",
                        value: "132 mg/dL"
                    },
                    {
                        time: "30 minutes",
                        value: "224 mg/dL"
                    },
                    {
                        time: "60 minutes",
                        value: "268 mg/dL"
                    },
                    {
                        time: "90 minutes",
                        value: "245 mg/dL"
                    },
                    {
                        time: "120 minutes (2 hours)",
                        value: "215 mg/dL"
                    }
                ],
                interpretation: "The Oral Glucose Tolerance Test confirms the diagnosis of Type 2 Diabetes Mellitus. Both the fasting glucose (132 mg/dL, diagnostic threshold ≥126 mg/dL) and the 2-hour post-glucose load value (215 mg/dL, diagnostic threshold ≥200 mg/dL) exceed the diagnostic criteria for diabetes. The elevated glucose levels throughout the test indicate significant impairment in glucose metabolism and insulin resistance.",
                recommendation: "Initiate treatment with Metformin 500mg twice daily with meals, gradually increasing to 1000mg twice daily as tolerated over 4 weeks. Comprehensive diabetes education including blood glucose monitoring, dietary management, and recognition of hypoglycemia symptoms. Regular physical activity as tolerated given osteoarthritis limitations. Close monitoring of renal function with initiation of Metformin. Recommend ophthalmology evaluation for baseline diabetic retinopathy screening."
            }
        },
        {
            year: 2013,
            age: 73,
            date: "October 24, 2013",
            vitals: {
                bloodPressure: "140/88",
                bloodPressureStatus: "warning",
                heartRate: "80",
                heartRateStatus: "normal",
                respiratoryRate: "18",
                respiratoryRateStatus: "normal",
                temperature: "36.6",
                temperatureStatus: "normal",
                weight: "77.5",
                weightStatus: "warning",
                height: "171",
                bmi: "26.5",
                bmiStatus: "warning"
            },
            bloodWorkResults: {
                hemoglobin: "13.5",
                hemoglobinStatus: "normal",
                whiteBloodCells: "7.6",
                whiteBloodCellsStatus: "normal",
                platelets: "218",
                plateletsStatus: "normal",
                totalCholesterol: "192",
                totalCholesterolStatus: "normal",
                ldlCholesterol: "116",
                ldlCholesterolStatus: "warning",
                hdlCholesterol: "42",
                hdlCholesterolStatus: "warning",
                triglycerides: "182",
                triglyceridesStatus: "warning",
                fastingGlucose: "135",
                fastingGlucoseStatus: "abnormal",
                hba1c: "6.8",
                hba1cStatus: "abnormal",
                bun: "19",
                bunStatus: "normal",
                creatinine: "1.02",
                creatinineStatus: "normal",
                egfr: "76",
                egfrStatus: "normal",
                microalbumin: "28",
                microalbuminStatus: "warning"
            },
            imagingResults: {
                chestXray: "Stable compared to previous. No acute cardiopulmonary disease.",
                chestXrayStatus: "warning"
            },
            additionalNotes: "Patient managing type 2 diabetes with Metformin 1000mg twice daily. Reports occasional mild gastrointestinal discomfort with medication. Currently also on Lisinopril 20mg daily (increased dose) and atorvastatin 20mg daily. Early signs of diabetic nephropathy with mildly elevated microalbumin. Diabetic retinopathy screening negative. Complains of increased joint pain, particularly in knees and right hip, limiting physical activity.",
            recommendedFollowUp: "Continue current management. Consider additional diabetes medication if HbA1c continues to rise. Nephrology referral for early diabetic nephropathy. Orthopedic consultation for worsening joint pain.",
            diagnosis: [
                {
                    name: "Essential Hypertension",
                    code: "I10",
                    status: "controlled",
                    notes: "Managed with ACE inhibitor, increased to 20mg daily"
                },
                {
                    name: "Type 2 Diabetes Mellitus",
                    code: "E11.9",
                    status: "ongoing",
                    notes: "HbA1c 6.8%, Managed with Metformin"
                },
                {
                    name: "Early Diabetic Nephropathy",
                    code: "E11.21",
                    status: "newly diagnosed",
                    notes: "Microalbuminuria present"
                }
            ]
        },
        {
            year: 2015,
            age: 75,
            date: "October 21, 2015",
            vitals: {
                bloodPressure: "136/84",
                bloodPressureStatus: "warning",
                heartRate: "82",
                heartRateStatus: "normal",
                respiratoryRate: "19",
                respiratoryRateStatus: "normal",
                temperature: "36.7",
                temperatureStatus: "normal",
                weight: "76.8",
                weightStatus: "warning",
                height: "170.5",
                bmi: "26.4",
                bmiStatus: "warning"
            },
            bloodWorkResults: {
                hemoglobin: "13.3",
                hemoglobinStatus: "normal",
                whiteBloodCells: "7.8",
                whiteBloodCellsStatus: "normal",
                platelets: "210",
                plateletsStatus: "normal",
                totalCholesterol: "188",
                totalCholesterolStatus: "normal",
                ldlCholesterol: "114",
                ldlCholesterolStatus: "warning",
                hdlCholesterol: "40",
                hdlCholesterolStatus: "warning",
                triglycerides: "195",
                triglyceridesStatus: "warning",
                fastingGlucose: "142",
                fastingGlucoseStatus: "abnormal",
                hba1c: "7.2",
                hba1cStatus: "abnormal",
                bun: "21",
                bunStatus: "normal",
                creatinine: "1.08",
                creatinineStatus: "normal",
                egfr: "71",
                egfrStatus: "warning",
                microalbumin: "45",
                microalbuminStatus: "abnormal"
            },
            imagingResults: {
                chestXray: "Mild cardiomegaly. No acute pulmonary disease. Stable aortic calcification.",
                chestXrayStatus: "warning",
                boneDensitometry: "Lumbar spine T-score: -1.8, Femoral neck T-score: -2.1. Findings consistent with osteopenia, approaching osteoporosis at femoral neck.",
                boneDensitometryStatus: "warning"
            },
            additionalNotes: "Patient experienced a fall at home 3 months ago without serious injury. Bone densitometry shows osteopenia/early osteoporosis. Diabetes management now includes Metformin 1000mg twice daily and recently added Sitagliptin 100mg daily. Hypertension stable on current regimen. Early diabetic nephropathy progressing with higher microalbumin levels. Patient reports increased general fatigue and some dizziness upon standing quickly.",
            recommendedFollowUp: "Start calcium and vitamin D supplementation. Consider bisphosphonate therapy for osteoporosis prevention. Adjust diabetes medication regimen. Home safety evaluation recommended to prevent falls.",
            diagnosis: [
                {
                    name: "Essential Hypertension",
                    code: "I10",
                    status: "controlled",
                    notes: "Stable on current medication"
                },
                {
                    name: "Type 2 Diabetes Mellitus",
                    code: "E11.9",
                    status: "ongoing",
                    notes: "HbA1c 7.2%, Added second oral agent"
                },
                {
                    name: "Diabetic Nephropathy",
                    code: "E11.21",
                    status: "worsening",
                    notes: "Increasing microalbuminuria"
                },
                {
                    name: "Osteopenia/Early Osteoporosis",
                    code: "M81.0",
                    status: "newly diagnosed",
                    notes: "Femoral neck T-score: -2.1"
                }
            ]
        },
        {
            year: 2017,
            age: 77,
            date: "October 18, 2017",
            vitals: {
                bloodPressure: "145/85",
                bloodPressureStatus: "abnormal",
                heartRate: "84",
                heartRateStatus: "normal",
                respiratoryRate: "20",
                respiratoryRateStatus: "normal",
                temperature: "36.5",
                temperatureStatus: "normal",
                weight: "75.2",
                weightStatus: "warning",
                height: "170",
                bmi: "26.0",
                bmiStatus: "warning"
            },
            bloodWorkResults: {
                hemoglobin: "13.0",
                hemoglobinStatus: "normal",
                whiteBloodCells: "8.0",
                whiteBloodCellsStatus: "normal",
                platelets: "205",
                plateletsStatus: "normal",
                totalCholesterol: "182",
                totalCholesterolStatus: "normal",
                ldlCholesterol: "110",
                ldlCholesterolStatus: "warning",
                hdlCholesterol: "38",
                hdlCholesterolStatus: "abnormal",
                triglycerides: "210",
                triglyceridesStatus: "warning",
                fastingGlucose: "156",
                fastingGlucoseStatus: "abnormal",
                hba1c: "7.5",
                hba1cStatus: "abnormal",
                bun: "24",
                bunStatus: "warning",
                creatinine: "1.15",
                creatinineStatus: "warning",
                egfr: "65",
                egfrStatus: "warning",
                microalbumin: "68",
                microalbuminStatus: "abnormal"
            },
            imagingResults: {
                chestXray: "Stable mild cardiomegaly. No acute pulmonary disease.",
                chestXrayStatus: "warning",
                carotidUltrasound: "Bilateral carotid atherosclerosis with 30-40% stenosis of right internal carotid artery and 20-30% stenosis of left internal carotid artery. No hemodynamically significant stenosis.",
                carotidUltrasoundStatus: "warning"
            },
            additionalNotes: "Blood pressure control has become more challenging. Patient reports episodes of dizziness when standing, suggestive of orthostatic hypotension. Diabetes control suboptimal despite dual oral therapy. Kidney function showing gradual decline. Memory difficulties reported by family members, particularly with short-term memory. Carotid ultrasound shows moderate atherosclerosis but no intervention required at this time.",
            recommendedFollowUp: "Adjust hypertension medication with careful consideration of orthostatic hypotension risk. Consider insulin therapy for improved glycemic control. Continue nephrology follow-up. Cognitive assessment recommended.",
            diagnosis: [
                {
                    name: "Essential Hypertension",
                    code: "I10",
                    status: "worsening",
                    notes: "More difficult to control, complicated by orthostatic hypotension"
                },
                {
                    name: "Type 2 Diabetes Mellitus",
                    code: "E11.9",
                    status: "worsening",
                    notes: "HbA1c 7.5%, May require insulin"
                },
                {
                    name: "Diabetic Nephropathy",
                    code: "E11.22",
                    status: "worsening",
                    notes: "Declining eGFR and increasing albuminuria"
                },
                {
                    name: "Carotid Atherosclerosis",
                    code: "I70.1",
                    status: "newly diagnosed",
                    notes: "Moderate atherosclerosis without hemodynamically significant stenosis"
                },
                {
                    name: "Mild Cognitive Impairment",
                    code: "G31.84",
                    status: "suspected",
                    notes: "Requires formal cognitive assessment"
                }
            ]
        },
        {
            year: 2019,
            age: 79,
            date: "October 23, 2019",
            vitals: {
                bloodPressure: "148/88",
                bloodPressureStatus: "abnormal",
                heartRate: "86",
                heartRateStatus: "normal",
                respiratoryRate: "20",
                respiratoryRateStatus: "normal",
                temperature: "36.6",
                temperatureStatus: "normal",
                weight: "73.8",
                weightStatus: "normal",
                height: "169.5",
                bmi: "25.7",
                bmiStatus: "warning"
            },
            bloodWorkResults: {
                hemoglobin: "12.8",
                hemoglobinStatus: "normal",
                whiteBloodCells: "8.2",
                whiteBloodCellsStatus: "normal",
                platelets: "198",
                plateletsStatus: "normal",
                totalCholesterol: "176",
                totalCholesterolStatus: "normal",
                ldlCholesterol: "106",
                ldlCholesterolStatus: "warning",
                hdlCholesterol: "36",
                hdlCholesterolStatus: "abnormal",
                triglycerides: "225",
                triglyceridesStatus: "warning",
                fastingGlucose: "165",
                fastingGlucoseStatus: "abnormal",
                hba1c: "7.8",
                hba1cStatus: "abnormal",
                bun: "27",
                bunStatus: "warning",
                creatinine: "1.25",
                creatinineStatus: "warning",
                egfr: "58",
                egfrStatus: "abnormal",
                microalbumin: "82",
                microalbuminStatus: "abnormal"
            },
            imagingResults: {
                chestXray: "Stable mild cardiomegaly. No acute pulmonary disease.",
                chestXrayStatus: "warning",
                brainMRI: "Moderate generalized cerebral atrophy consistent with age. Multiple small chronic ischemic changes in periventricular and subcortical white matter, consistent with small vessel disease. No acute intracranial abnormalities.",
                brainMRIStatus: "warning"
            },
            additionalNotes: "Patient now on insulin therapy (Insulin glargine 18 units at bedtime) in addition to Metformin (reduced to 500mg twice daily due to declining kidney function). Cognitive assessment confirms mild cognitive impairment. Brain MRI shows evidence of small vessel disease, likely contributing to cognitive changes. Hypertension management complicated by competing risks of inadequate cerebral perfusion vs. target organ damage. Patient reports decreased mobility due to joint pain and occasional balance problems.",
            recommendedFollowUp: "Geriatrics consultation for comprehensive management of multiple chronic conditions. Nephrology follow-up for advancing kidney disease. Memory clinic referral for cognitive support strategies.",
            diagnosis: [
                {
                    name: "Essential Hypertension",
                    code: "I10",
                    status: "ongoing",
                    notes: "Target adjusted to <150/90 mmHg given age and comorbidities"
                },
                {
                    name: "Type 2 Diabetes Mellitus",
                    code: "E11.9",
                    status: "ongoing",
                    notes: "HbA1c 7.8%, Now requiring insulin"
                },
                {
                    name: "Diabetic Nephropathy",
                    code: "E11.22",
                    status: "worsening",
                    notes: "Stage 3 Chronic Kidney Disease"
                },
                {
                    name: "Mild Cognitive Impairment",
                    code: "G31.84",
                    status: "confirmed",
                    notes: "Consistent with small vessel disease on imaging"
                },
                {
                    name: "Cerebral Small Vessel Disease",
                    code: "I67.81",
                    status: "newly diagnosed",
                    notes: "Evident on brain MRI"
                }
            ]
        },
        {
            year: 2021,
            age: 81,
            date: "October 20, 2021",
            vitals: {
                bloodPressure: "145/80",
                bloodPressureStatus: "warning",
                heartRate: "88",
                heartRateStatus: "normal",
                respiratoryRate: "21",
                respiratoryRateStatus: "normal",
                temperature: "36.4",
                temperatureStatus: "normal",
                weight: "72.1",
                weightStatus: "normal",
                height: "169",
                bmi: "25.2",
                bmiStatus: "warning"
            },
            bloodWorkResults: {
                hemoglobin: "12.5",
                hemoglobinStatus: "warning",
                whiteBloodCells: "8.5",
                whiteBloodCellsStatus: "normal",
                platelets: "192",
                plateletsStatus: "normal",
                totalCholesterol: "168",
                totalCholesterolStatus: "normal",
                ldlCholesterol: "102",
                ldlCholesterolStatus: "warning",
                hdlCholesterol: "35",
                hdlCholesterolStatus: "abnormal",
                triglycerides: "235",
                triglyceridesStatus: "warning",
                fastingGlucose: "172",
                fastingGlucoseStatus: "abnormal",
                hba1c: "7.9",
                hba1cStatus: "abnormal",
                bun: "32",
                bunStatus: "abnormal",
                creatinine: "1.38",
                creatinineStatus: "abnormal",
                egfr: "50",
                egfrStatus: "abnormal",
                microalbumin: "105",
                microalbuminStatus: "abnormal"
            },
            imagingResults: {
                chestXray: "Stable mild cardiomegaly. No acute pulmonary disease.",
                chestXrayStatus: "warning",
                echocardiogram: "Left ventricular ejection fraction 54%. Moderate left ventricular hypertrophy. Mild aortic stenosis. Grade I diastolic dysfunction.",
                echocardiogramStatus: "warning"
            },
            additionalNotes: "Patient experienced a transient ischemic attack 3 months ago with temporary right-sided weakness that resolved within 24 hours. Now on antiplatelet therapy (aspirin 100mg daily). Cognitive decline has progressed moderately, affecting instrumental activities of daily living. Diabetes management includes insulin glargine 24 units at bedtime and Metformin 500mg daily. Chronic kidney disease progressing. Medication regimen simplified to improve adherence. Family reports patient sometimes forgets to take medications.",
            recommendedFollowUp: "Consider medication dispensing system or family assistance with medication management. Continued monitoring of kidney function. Adjust diabetes management goals to prioritize safety and avoid hypoglycemia.",
            diagnosis: [
                {
                    name: "Essential Hypertension",
                    code: "I10",
                    status: "ongoing",
                    notes: "Target remains <150/90 mmHg given age and comorbidities"
                },
                {
                    name: "Type 2 Diabetes Mellitus",
                    code: "E11.9",
                    status: "ongoing",
                    notes: "HbA1c 7.9%, Goal adjusted to <8.0%"
                },
                {
                    name: "Diabetic Nephropathy",
                    code: "E11.22",
                    status: "worsening",
                    notes: "Stage 3b Chronic Kidney Disease"
                },
                {
                    name: "Mild Cognitive Impairment",
                    code: "G31.84",
                    status: "worsening",
                    notes: "Affecting instrumental activities of daily living"
                },
                {
                    name: "Cerebral Small Vessel Disease",
                    code: "I67.81",
                    status: "ongoing",
                    notes: "Recent transient ischemic attack"
                },
                {
                    name: "Transient Ischemic Attack",
                    code: "G45.9",
                    status: "newly diagnosed",
                    notes: "3 months ago, now on antiplatelet therapy"
                }
            ]
        },
        {
            year: 2023,
            age: 83,
            date: "October 19, 2023",
            vitals: {
                bloodPressure: "148/82",
                bloodPressureStatus: "warning",
                heartRate: "90",
                heartRateStatus: "normal",
                respiratoryRate: "22",
                respiratoryRateStatus: "normal",
                temperature: "36.3",
                temperatureStatus: "normal",
                weight: "70.4",
                weightStatus: "normal",
                height: "168.5",
                bmi: "24.8",
                bmiStatus: "normal"
            },
            bloodWorkResults: {
                hemoglobin: "12.2",
                hemoglobinStatus: "warning",
                whiteBloodCells: "8.8",
                whiteBloodCellsStatus: "normal",
                platelets: "185",
                plateletsStatus: "normal",
                totalCholesterol: "162",
                totalCholesterolStatus: "normal",
                ldlCholesterol: "98",
                ldlCholesterolStatus: "normal",
                hdlCholesterol: "34",
                hdlCholesterolStatus: "abnormal",
                triglycerides: "245",
                triglyceridesStatus: "abnormal",
                fastingGlucose: "168",
                fastingGlucoseStatus: "abnormal",
                hba1c: "8.0",
                hba1cStatus: "abnormal",
                bun: "38",
                bunStatus: "abnormal",
                creatinine: "1.52",
                creatinineStatus: "abnormal",
                egfr: "44",
                egfrStatus: "abnormal",
                microalbumin: "128",
                microalbuminStatus: "abnormal"
            },
            imagingResults: {
                chestXray: "Stable mild cardiomegaly. Early signs of pulmonary congestion. No acute consolidation.",
                chestXrayStatus: "warning",
                abdominalUltrasound: "Multiple gallstones without evidence of cholecystitis. Moderate hepatic steatosis. Kidneys show decreased size and increased echogenicity consistent with chronic kidney disease.",
                abdominalUltrasoundStatus: "warning"
            },
            additionalNotes: "Patient more frail with significant functional limitations. Requires assistance with some basic activities of daily living. Cognitive impairment has progressed further. Medication regimen substantially simplified to focus on symptom management and quality of life. Diabetes management focused primarily on avoiding hypoglycemia and extreme hyperglycemia rather than tight control. Brief hospitalization 2 months ago for heart failure exacerbation with good response to diuretic therapy.",
            recommendedFollowUp: "Coordinate care with geriatrics, cardiology, and nephrology. Consider palliative care consultation for symptom management and advance care planning. Adjust medication regimen for safety and quality of life.",
            diagnosis: [
                {
                    name: "Essential Hypertension",
                    code: "I10",
                    status: "ongoing",
                    notes: "Target remains <150/90 mmHg given age and comorbidities"
                },
                {
                    name: "Type 2 Diabetes Mellitus",
                    code: "E11.9",
                    status: "ongoing",
                    notes: "HbA1c 8.0%, Goal adjusted to <8.5%"
                },
                {
                    name: "Diabetic Nephropathy",
                    code: "E11.22",
                    status: "worsening",
                    notes: "Stage 3b Chronic Kidney Disease"
                },
                {
                    name: "Cognitive Impairment",
                    code: "G31.84",
                    status: "worsening",
                    notes: "Now affecting basic activities of daily living"
                },
                {
                    name: "Heart Failure with Preserved Ejection Fraction",
                    code: "I50.3",
                    status: "newly diagnosed",
                    notes: "Recent hospitalization for exacerbation"
                },
                {
                    name: "Cholelithiasis",
                    code: "K80.20",
                    status: "newly diagnosed",
                    notes: "Asymptomatic, incidental finding"
                },
                {
                    name: "Hepatic Steatosis",
                    code: "K76.0",
                    status: "newly diagnosed",
                    notes: "Moderate severity on ultrasound"
                }
            ]
        }
    ];

    // Extra diagnostic test details
    const diagnosticTests = {
        "Oral Glucose Tolerance Test (OGTT)": {
            purpose: "To evaluate how the body processes glucose, used for diagnosing diabetes, prediabetes, and other conditions affecting glucose metabolism.",
            procedure: "Patient fasts for at least 8 hours, then has blood drawn for baseline glucose measurement. Then drinks a solution containing 75 grams of glucose. Blood samples are drawn at timed intervals (typically 30, 60, 90, and 120 minutes) to measure how glucose levels change.",
            interpretation: "For diabetes diagnosis: fasting glucose ≥126 mg/dL or 2-hour glucose ≥200 mg/dL.",
            advantages: "Provides comprehensive information about glucose metabolism; can detect diabetes earlier than fasting glucose alone; shows pattern of glucose handling.",
            limitations: "Time-consuming (takes several hours); requires multiple blood draws; affected by recent illness, medications, and stress; less convenient than HbA1c testing."
        },
        "24-hour Ambulatory Blood Pressure Monitoring": {
            purpose: "To measure blood pressure at regular intervals over a 24-hour period during normal daily activities and sleep to diagnose hypertension and evaluate BP patterns.",
            procedure: "A portable blood pressure measuring device is worn with a cuff attached to the upper arm, connected to a small monitoring device usually worn on a belt. The device takes BP readings automatically every 15-30 minutes during the day and every 30-60 minutes during sleep.",
            interpretation: "Daytime average >135/85 mmHg or 24-hour average >130/80 mmHg typically indicates hypertension. The absence of a normal nighttime dip in blood pressure (non-dipping pattern) may indicate increased cardiovascular risk.",
            advantages: "Provides multiple readings in the patient's natural environment; identifies white-coat hypertension and masked hypertension; assesses blood pressure variability and circadian rhythm; better predictor of cardiovascular risk than office measurements.",
            limitations: "Device may disturb sleep and normal activities; occasional measurement failures; may cause discomfort; requires patient compliance; more expensive than office BP measurement."
        }
    };

    // Function to get class for status
    function getStatusClass(status) {
        switch(status) {
            case 'normal': return 'normal';
            case 'warning': return 'warning';
            case 'abnormal': return 'abnormal';
            default: return 'normal';
        }
    }

    // Function to calculate the percentage for progress bars
    function calculatePercentage(value, min, max, inverse = false) {
        let percentage = ((value - min) / (max - min)) * 100;
        percentage = Math.min(Math.max(percentage, 0), 100);
        return inverse ? 100 - percentage : percentage;
    }

    // Generate reports
    function generateReports() {
        const reportsContainer = document.getElementById('reports-container');
        reportsContainer.innerHTML = '';

        medicalReports.forEach(report => {
            const reportCard = document.createElement('div');
            reportCard.className = 'report-card relative bg-white rounded-lg shadow-sm border border-gray-200 p-5 data-type-all';
            if (report.diagnosis.some(d => d.status === "newly diagnosed")) {
                reportCard.classList.add('data-type-abnormal');
            }
            if (report.vitals.bloodPressureStatus !== "normal") {
                reportCard.classList.add('data-type-blood-pressure');
            }
            if (report.bloodWorkResults.fastingGlucoseStatus !== "normal" || 
                (report.bloodWorkResults.hba1cStatus && report.bloodWorkResults.hba1cStatus !== "normal")) {
                reportCard.classList.add('data-type-glucose');
            }

            // Year badge
            const yearBadge = document.createElement('div');
            yearBadge.className = 'year-badge';
            yearBadge.innerHTML = `
                <div class="text-center">
                    <div class="text-sm font-bold">${report.year}</div>
                    <div class="patient-age">${report.age}岁</div>
                </div>
            `;
            reportCard.appendChild(yearBadge);

            // Main content
            reportCard.innerHTML += `
                <div class="mb-4 pl-10">
                    <h3 class="text-lg font-semibold">Medical Examination Report</h3>
                    <p class="text-sm text-gray-500">Date: ${report.date}</p>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <h4 class="text-md font-medium mb-2 flex items-center">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                            </svg>
                            Vital Signs
                        </h4>
                        <div class="space-y-3 text-sm">
                            <div class="data-point ${getStatusClass(report.vitals.bloodPressureStatus)}">
                                <div class="flex justify-between mb-1">
                                    <span>Blood Pressure</span>
                                    <span class="test-result-value ${report.vitals.bloodPressureStatus !== 'normal' ? `value-${report.vitals.bloodPressureStatus}` : ''}">${report.vitals.bloodPressure} mmHg</span>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-bar-fill ${getStatusClass(report.vitals.bloodPressureStatus)}" style="width: ${calculatePercentage(parseInt(report.vitals.bloodPressure.split('/')[0]), 100, 180)}%"></div>
                                </div>
                            </div>
                            
                            <div class="data-point ${getStatusClass(report.vitals.heartRateStatus)}">
                                <div class="flex justify-between mb-1">
                                    <span>Heart Rate</span>
                                    <span class="test-result-value ${report.vitals.heartRateStatus !== 'normal' ? `value-${report.vitals.heartRateStatus}` : ''}">${report.vitals.heartRate} bpm</span>
                                </div>
                            </div>
                            
                            <div class="data-point ${getStatusClass(report.vitals.weightStatus)}">
                                <div class="flex justify-between mb-1">
                                    <span>Weight</span>
                                    <span class="test-result-value ${report.vitals.weightStatus !== 'normal' ? `value-${report.vitals.weightStatus}` : ''}">${report.vitals.weight} kg</span>
                                </div>
                            </div>
                            
                            <div class="data-point ${getStatusClass(report.vitals.bmiStatus)}">
                                <div class="flex justify-between mb-1">
                                    <span>BMI</span>
                                    <span class="test-result-value ${report.vitals.bmiStatus !== 'normal' ? `value-${report.vitals.bmiStatus}` : ''}">${report.vitals.bmi} kg/m²</span>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-bar-fill ${getStatusClass(report.vitals.bmiStatus)}" style="width: ${calculatePercentage(parseFloat(report.vitals.bmi), 18.5, 30)}%"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div>
                        <h4 class="text-md font-medium mb-2 flex items-center">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                            </svg>
                            Blood Work Results
                        </h4>
                        <div class="space-y-3 text-sm">
                            <div class="data-point ${getStatusClass(report.bloodWorkResults.fastingGlucoseStatus)}">
                                <div class="flex justify-between mb-1">
                                    <span>Fasting Glucose</span>
                                    <span class="test-result-value ${report.bloodWorkResults.fastingGlucoseStatus !== 'normal' ? `value-${report.bloodWorkResults.fastingGlucoseStatus}` : ''}">${report.bloodWorkResults.fastingGlucose} mg/dL</span>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-bar-fill ${getStatusClass(report.bloodWorkResults.fastingGlucoseStatus)}" style="width: ${calculatePercentage(parseInt(report.bloodWorkResults.fastingGlucose), 70, 200)}%"></div>
                                </div>
                            </div>
                            
                            <div class="data-point ${getStatusClass(report.bloodWorkResults.hba1cStatus)}">
                                <div class="flex justify-between mb-1">
                                    <span>HbA1c</span>
                                    <span class="test-result-value ${report.bloodWorkResults.hba1cStatus !== 'normal' ? `value-${report.bloodWorkResults.hba1cStatus}` : ''}">${report.bloodWorkResults.hba1c}%</span>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-bar-fill ${getStatusClass(report.bloodWorkResults.hba1cStatus)}" style="width: ${calculatePercentage(parseFloat(report.bloodWorkResults.hba1c), 4.0, 9.0)}%"></div>
                                </div>
                            </div>
                            
                            <div class="data-point ${getStatusClass(report.bloodWorkResults.totalCholesterolStatus)}">
                                <div class="flex justify-between mb-1">
                                    <span>Total Cholesterol</span>
                                    <span class="test-result-value ${report.bloodWorkResults.totalCholesterolStatus !== 'normal' ? `value-${report.bloodWorkResults.totalCholesterolStatus}` : ''}">${report.bloodWorkResults.totalCholesterol} mg/dL</span>
                                </div>
                            </div>
                            
                            <div class="data-point ${getStatusClass(report.bloodWorkResults.creatinineStatus)}">
                                <div class="flex justify-between mb-1">
                                    <span>Creatinine</span>
                                    <span class="test-result-value ${report.bloodWorkResults.creatinineStatus !== 'normal' ? `value-${report.bloodWorkResults.creatinineStatus}` : ''}">${report.bloodWorkResults.creatinine} mg/dL</span>
                                </div>
                            </div>
                            
                            ${report.bloodWorkResults.microalbumin ? `
                            <div class="data-point ${getStatusClass(report.bloodWorkResults.microalbuminStatus)}">
                                <div class="flex justify-between mb-1">
                                    <span>Microalbumin</span>
                                    <span class="test-result-value ${report.bloodWorkResults.microalbuminStatus !== 'normal' ? `value-${report.bloodWorkResults.microalbuminStatus}` : ''}">${report.bloodWorkResults.microalbumin} mg/L</span>
                                </div>
                            </div>
                            ` : ''}
                        </div>
                    </div>
                </div>

                ${report.diagnosis.length > 0 ? `
                <div class="mt-5">
                    <h4 class="text-md font-medium mb-2 flex items-center">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        Diagnosis
                    </h4>
                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-2">
                        ${report.diagnosis.map(diagnosis => `
                            <div class="bg-gray-50 rounded p-3 relative">
                                ${diagnosis.status === "newly diagnosed" ? `<span class="diagnostic-tag added absolute top-2 right-2">New</span>` : ''}
                                <div class="font-medium">${diagnosis.name}</div>
                                <div class="text-sm text-gray-600 mt-1">${diagnosis.notes}</div>
                                ${diagnosis.confirmatoryTests ? `
                                <div class="mt-2">
                                    <button class="test-detail-btn text-xs text-blue-600 flex items-center" data-test="${diagnosis.confirmatoryTests}">
                                        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                        View ${diagnosis.confirmatoryTests} details
                                    </button>
                                </div>
                                ` : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>
                ` : ''}

                <div class="mt-5">
                    <h4 class="text-md font-medium mb-2 flex items-center">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                        </svg>
                        Clinical Notes
                    </h4>
                    <p class="text-sm text-gray-700 mt-2">${report.additionalNotes}</p>
                </div>
            `;
            
            reportsContainer.appendChild(reportCard);
        });

        // Add event listeners to diagnostic test detail buttons
        document.querySelectorAll('.test-detail-btn').forEach(button => {
            button.addEventListener('click', function() {
                const testName = this.getAttribute('data-test');
                const modal = document.getElementById('diagnosticModal');
                const modalTitle = document.getElementById('modalTitle');
                const modalContent = document.getElementById('modalContent');
                
                modalTitle.textContent = testName;
                
                // Find the report that has this diagnostic test
                const report = medicalReports.find(report => 
                    report.diagnosis && report.diagnosis.some(d => d.confirmatoryTests === testName)
                );
                
                if (report && report.diagnosticTests) {
                    const testData = report.diagnosticTests;
                    let content = `
                        <div class="space-y-4">
                            <div>
                                <h4 class="font-medium text-gray-900">Test Results (${testData.date})</h4>
                                <div class="mt-2 bg-gray-50 rounded-lg p-3">
                                    <table class="w-full text-sm">
                                        <thead>
                                            <tr>
                                                <th class="text-left font-medium text-gray-500 pb-2">Time Point</th>
                                                <th class="text-right font-medium text-gray-500 pb-2">Value</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            ${testData.results.map(result => `
                                                <tr>
                                                    <td class="py-1">${result.time}</td>
                                                    <td class="text-right font-medium">${result.value}</td>
                                                </tr>
                                            `).join('')}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                            <div>
                                <h4 class="font-medium text-gray-900">Interpretation</h4>
                                <p class="mt-1 text-gray-700">${testData.interpretation}</p>
                            </div>
                            <div>
                                <h4 class="font-medium text-gray-900">Recommendation</h4>
                                <p class="mt-1 text-gray-700">${testData.recommendation}</p>
                            </div>
                        </div>
                    `;
                    
                    // Add general test information
                    if (diagnosticTests[testName]) {
                        const generalInfo = diagnosticTests[testName];
                        content += `
                            <div class="mt-6 pt-6 border-t border-gray-200">
                                <h4 class="font-medium text-gray-900">About ${testName}</h4>
                                <div class="mt-3 space-y-3 text-sm">
                                    <div>
                                        <div class="font-medium text-gray-700">Purpose:</div>
                                        <p class="text-gray-600">${generalInfo.purpose}</p>
                                    </div>
                                    <div>
                                        <div class="font-medium text-gray-700">Procedure:</div>
                                        <p class="text-gray-600">${generalInfo.procedure}</p>
                                    </div>
                                    <div>
                                        <div class="font-medium text-gray-700">Interpretation Standards:</div>
                                        <p class="text-gray-600">${generalInfo.interpretation}</p>
                                    </div>
                                    <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                                        <div>
                                            <div class="font-medium text-gray-700">Advantages:</div>
                                            <p class="text-gray-600">${generalInfo.advantages}</p>
                                        </div>
                                        <div>
                                            <div class="font-medium text-gray-700">Limitations:</div>
                                            <p class="text-gray-600">${generalInfo.limitations}</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                    
                    modalContent.innerHTML = content;
                } else {
                    modalContent.innerHTML = `<p>Details for ${testName} are not available.</p>`;
                }
                
                modal.classList.remove('hidden');
            });
        });

        // Close modal button
        document.getElementById('closeModal').addEventListener('click', function() {
            document.getElementById('diagnosticModal').classList.add('hidden');
        });

        // Close modal when clicking outside
        document.getElementById('diagnosticModal').addEventListener('click', function(e) {
            if (e.target === this) {
                this.classList.add('hidden');
            }
        });
    }

    // Filter reports
    document.getElementById('filterReports').addEventListener('change', function() {
        const filterValue = this.value;
        const reports = document.querySelectorAll('.report-card');
        
        reports.forEach(report => {
            if (filterValue === 'all') {
                report.style.display = 'block';
            } else {
                report.style.display = report.classList.contains(`data-type-${filterValue}`) ? 'block' : 'none';
            }
        });
    });

    // Initialize reports
    generateReports();
});
