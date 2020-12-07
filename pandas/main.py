import random

randomseed = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG',
'AH','AI','AJ','AK','AL','AM','AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY','AZ','BA','BB','BC','BD','BE','BF','BG','BH','BI','BJ','BK','BL',
'BM','BN','BO','BP','BQ','BR','BS','BT','BU','BV','BW','BX','BY','BZ','CA','CB','CC','CD','CE','CF','CG','CH','CI','CJ','CK','CL','CM','CN','CO','CP','CQ',
'CR','CS','CT','CU','CV','CW','CX','CY','CZ','DA','DB','DC','DD','DE','DF','DG','DH','DI','DJ','DK','DL','DM','DN','DO','DP','DQ','DR','DS','DT','DU','DV',
'DW','DX','DY','DZ','EA','EB','EC','ED','EE','EF','EG','EH','EI','EJ','EK','EL','EM','EN','EO','EP','EQ','ER','ES','ET','EU','EV','EW','EX','EY','EZ','FA',
'FB','FC','FD','FE','FF','FG','FH','FI','FJ','FK','FL','FM','FN','FO','FP','FQ','FR','FS','FT','FU','FV','FW','FX','FY','FZ','GA','GB','GC','GD','GE','GF',
'GG','GH','GI','GJ','GK','GL','GM','GN','GO','GP','GQ','GR','GS','GT','GU','GV','GW','GX','GY','GZ','HA','HB','HC','HD','HE','HF','HG','HH','HI','HJ','HK',
'HL','HM','HN','HO','HP','HQ','HR','HS','HT','HU','HV','HW','HX','HY','HZ','IA','IB','IC','ID','IE','IF','IG','IH','II','IJ','IK','IL','IM','IN','IO','IP',
'IQ','IR','IS','IT','IU','IV','IW','IX','IY','IZ','JA','JB','JC','JD','JE','JF','JG','JH','JI','JJ','JK','JL','JM','JN','JO','JP','JQ','JR','JS','JT','JU',
'JV','JW','JX','JY','JZ','KA','KB','KC','KD','KE','KF','KG','KH','KI','KJ','KK','KL','KM','KN','KO','KP','KQ','KR','KS','KT','KU','KV','KW','KX','KY','KZ',
'LA','LB','LC','LD','LE','LF','LG','LH','LI','LJ','LK','LL','LM','LN','LO','LP','LQ','LR','LS','LT','LU','LV','LW','LX','LY','LZ','MA','MB','MC','MD','ME',
'MF','MG','MH','MI','MJ','MK','ML','MM','MN','MO','MP','MQ','MR','MS','MT','MU','MV','MW','MX','MY','MZ','NA','NB','NC','ND','NE','NF','NG','NH','NI','NJ',
'NK','NL','NM','NN','NO','NP','NQ','NR','NS','NT','NU','NV','NW','NX','NY','NZ','OA','OB','OC','OD','OE','OF','OG','OH','OI','OJ','OK','OL','OM','ON','OO',
'OP','OQ','OR','OS','OT','OU','OV','OW','OX','OY','OZ','PA','PB','PC','PD','PE','PF','PG','PH','PI','PJ','PK','PL','PM','PN','PO','PP','PQ','PR','PS','PT',
'PU','PV','PW','PX','PY','PZ','QA','QB','QC','QD','QE','QF','QG','QH','QI','QJ','QK','QL','QM','QN','QO','QP','QQ','QR','QS','QT','QU','QV','QW','QX','QY',
'QZ','RA','RB','RC','RD','RE','RF','RG','RH','RI','RJ','RK','RL','RM','RN','RO','RP','RQ','RR','RS','RT','RU','RV','RW','RX','RY','RZ','SA','SB','SC','SD',
'SE','SF','SG','SH','SI','SJ','SK','SL','SM','SN','SO','SP','SQ','SR','SS','ST','SU','SV','SW','SX','SY','SZ','TA','TB','TC','TD','TE','TF','TG','TH','TI',
'TJ','TK','TL','TM','TN','TO','TP','TQ','TR','TS','TT','TU','TV','TW','TX','TY','TZ','UA','UB','UC','UD','UE','UF','UG','UH','UI','UJ','UK','UL','UM','UN',
'UO','UP','UQ','UR','US','UT','UU','UV','UW','UX','UY','UZ','VA','VB','VC','VD','VE','VF','VG','VH','VI','VJ','VK','VL','VM','VN','VO','VP','VQ','VR','VS',
'VT','VU','VV','VW','VX','VY','VZ','WA','WB','WC','WD','WE','WF','WG','WH','WI','WJ','WK','WL','WM','WN','WO','WP','WQ','WR','WS','WT','WU','WV','WW','WX',
'WY','WZ','XA','XB','XC','XD','XE','XF','XG','XH','XI','XJ','XK','XL','XM','XN','XO','XP','XQ','XR','XS','XT','XU','XV','XW','XX','XY','XZ','YA','YB','YC',
'YD','YE','YF','YG','YH','YI','YJ','YK','YL','YM','YN','YO','YP','YQ','YR','YS','YT','YU','YV','YW','YX','YY','YZ','ZA','ZB','ZC','ZD','ZE','ZF','ZG','ZH',
'ZI','ZJ','ZK','ZL','ZM','ZN','ZO','ZP','ZQ','ZR','ZS','ZT','ZU','ZV','ZW','ZX','ZY','ZZ','AAA','AAB','AAC','AAD','AAE','AAF','AAG','AAH','AAI','AAJ','AAK',
'AAL','AAM','AAN','AAO','AAP','AAQ','AAR','AAS','AAT','AAU','AAV','AAW','AAX','AAY','AAZ','ABA','ABB','ABC','ABD','ABE','ABF','ABG','ABH','ABI','ABJ','ABK',
'ABL','ABM','ABN','ABO','ABP','ABQ','ABR','ABS','ABT','ABU','ABV','ABW','ABX','ABY','ABZ','ACA','ACB','ACC','ACD','ACE','ACF','ACG','ACH','ACI','ACJ','ACK',
'ACL','ACM','ACN','ACO','ACP','ACQ','ACR','ACS','ACT','ACU','ACV','ACW','ACX','ACY','ACZ','ADA','ADB','ADC','ADD','ADE','ADF','ADG','ADH','ADI','ADJ','ADK',
'ADL','ADM','ADN','ADO','ADP','ADQ','ADR','ADS','ADT','ADU','ADV','ADW','ADX','ADY','ADZ','AEA','AEB','AEC','AED','AEE','AEF','AEG','AEH','AEI','AEJ','AEK',
'AEL','AEM','AEN','AEO','AEP','AEQ','AER','AES','AET','AEU','AEV','AEW','AEX','AEY','AEZ','AFA','AFB','AFC','AFD','AFE','AFF','AFG','AFH','AFI','AFJ','AFK',
'AFL','AFM','AFN','AFO','AFP','AFQ','AFR','AFS','AFT','AFU','AFV','AFW','AFX','AFY','AFZ','AGA','AGB','AGC','AGD','AGE','AGF','AGG','AGH','AGI','AGJ','AGK',
'AGL','AGM','AGN','AGO','AGP','AGQ','AGR','AGS','AGT','AGU','AGV','AGW','AGX','AGY','AGZ','AHA','AHB','AHC','AHD','AHE','AHF','AHG','AHH','AHI','AHJ','AHK',
'AHL','AHM','AHN','AHO','AHP','AHQ','AHR','AHS','AHT','AHU','AHV','AHW','AHX','AHY','AHZ','AIA','AIB','AIC','AID','AIE','AIF','AIG','AIH','AII','AIJ','AIK',
'AIL','AIM','AIN','AIO','AIP','AIQ','AIR','AIS','AIT','AIU','AIV','AIW','AIX','AIY','AIZ','AJA','AJB','AJC','AJD','AJE','AJF','AJG','AJH','AJI','AJJ','AJK',
'AJL','AJM','AJN','AJO','AJP','AJQ','AJR','AJS','AJT','AJU','AJV','AJW','AJX','AJY','AJZ','AKA','AKB','AKC','AKD','AKE','AKF','AKG','AKH','AKI','AKJ','AKK',
'AKL','AKM','AKN','AKO','AKP','AKQ','AKR','AKS','AKT','AKU','AKV','AKW','AKX','AKY','AKZ','ALA','ALB','ALC','ALD','ALE','ALF','ALG','ALH','ALI','ALJ','ALK',
'ALL','ALM','ALN','ALO','ALP','ALQ','ALR','ALS','ALT','ALU','ALV','ALW','ALX','ALY','ALZ','AMA','AMB','AMC','AMD','AME','AMF','AMG','AMH','AMI','AMJ','AMK',
'AML','AMM','AMN','AMO','AMP','AMQ','AMR','AMS','AMT','AMU','AMV','AMW','AMX','AMY','AMZ','ANA','ANB','ANC','AND','ANE','ANF','ANG','ANH','ANI','ANJ','ANK',
'ANL','ANM','ANN','ANO','ANP','ANQ','ANR','ANS','ANT','ANU','ANV','ANW','ANX','ANY','ANZ','AOA','AOB','AOC','AOD','AOE','AOF','AOG','AOH','AOI','AOJ','AOK',
'AOL','AOM','AON','AOO','AOP','AOQ','AOR','AOS','AOT','AOU','AOV','AOW','AOX','AOY','AOZ','APA','APB','APC','APD','APE','APF','APG','APH','API','APJ','APK',
'APL','APM','APN','APO','APP','APQ','APR','APS','APT','APU','APV','APW','APX','APY','APZ','AQA','AQB','AQC','AQD','AQE','AQF','AQG','AQH','AQI','AQJ','AQK',
'AQL','AQM','AQN','AQO','AQP','AQQ','AQR','AQS','AQT','AQU','AQV','AQW','AQX','AQY','AQZ','ARA','ARB','ARC','ARD','ARE','ARF','ARG','ARH','ARI','ARJ','ARK',
'ARL','ARM','ARN','ARO','ARP','ARQ','ARR','ARS','ART','ARU','ARV','ARW','ARX','ARY','ARZ','ASA','ASB','ASC','ASD','ASE','ASF','ASG','ASH','ASI','ASJ','ASK',
'ASL','ASM','ASN','ASO','ASP','ASQ','ASR','ASS','AST','ASU','ASV','ASW','ASX','ASY','ASZ','ATA','ATB','ATC','ATD','ATE','ATF','ATG','ATH','ATI','ATJ','ATK',
'ATL','ATM','ATN','ATO','ATP','ATQ','ATR','ATS','ATT','ATU','ATV','ATW','ATX','ATY','ATZ','AUA','AUB','AUC','AUD','AUE','AUF','AUG','AUH','AUI','AUJ','AUK',
'AUL','AUM','AUN','AUO','AUP','AUQ','AUR','AUS','AUT','AUU','AUV','AUW','AUX','AUY','AUZ','AVA','AVB','AVC','AVD','AVE','AVF','AVG','AVH','AVI','AVJ','AVK',
'AVL','AVM','AVN','AVO','AVP','AVQ','AVR','AVS','AVT','AVU','AVV','AVW','AVX','AVY','AVZ','AWA','AWB','AWC','AWD','AWE','AWF','AWG','AWH','AWI','AWJ','AWK',
'AWL','AWM','AWN','AWO','AWP','AWQ','AWR','AWS','AWT','AWU','AWV','AWW','AWX','AWY','AWZ','AXA','AXB','AXC','AXD','AXE','AXF','AXG','AXH','AXI','AXJ','AXK',
'AXL','AXM','AXN','AXO','AXP','AXQ','AXR','AXS','AXT','AXU','AXV','AXW','AXX','AXY','AXZ','AYA','AYB','AYC','AYD','AYE','AYF','AYG','AYH','AYI','AYJ','AYK',
'AYL','AYM','AYN','AYO','AYP','AYQ','AYR','AYS','AYT','AYU','AYV','AYW','AYX','AYY','AYZ','AZA','AZB','AZC','AZD','AZE','AZF','AZG','AZH','AZI','AZJ','AZK',
'AZL','AZM','AZN','AZO','AZP','AZQ','AZR','AZS','AZT','AZU','AZV','AZW','AZX','AZY','AZZ','BAA','BAB','BAC','BAD','BAE','BAF','BAG','BAH','BAI','BAJ','BAK',
'BAL','BAM','BAN','BAO','BAP','BAQ','BAR','BAS','BAT','BAU','BAV','BAW','BAX','BAY','BAZ','BBA','BBB','BBC','BBD','BBE','BBF','BBG','BBH','BBI','BBJ','BBK',
'BBL','BBM','BBN','BBO','BBP','BBQ','BBR','BBS','BBT','BBU','BBV','BBW','BBX','BBY','BBZ','BCA','BCB','BCC','BCD','BCE','BCF','BCG','BCH','BCI','BCJ','BCK',
'BCL','BCM','BCN','BCO','BCP','BCQ','BCR','BCS','BCT','BCU','BCV','BCW','BCX','BCY','BCZ','BDA','BDB','BDC','BDD','BDE','BDF','BDG','BDH','BDI','BDJ','BDK',
'BDL','BDM','BDN','BDO','BDP','BDQ','BDR','BDS','BDT','BDU','BDV','BDW','BDX','BDY','BDZ','BEA','BEB','BEC','BED','BEE','BEF','BEG','BEH','BEI','BEJ','BEK',
'BEL','BEM','BEN','BEO','BEP','BEQ','BER','BES','BET','BEU','BEV','BEW','BEX','BEY','BEZ','BFA','BFB','BFC','BFD','BFE','BFF','BFG','BFH','BFI','BFJ','BFK',
'BFL','BFM','BFN','BFO','BFP','BFQ','BFR','BFS','BFT','BFU','BFV','BFW','BFX','BFY','BFZ','BGA','BGB','BGC','BGD','BGE','BGF','BGG','BGH','BGI','BGJ','BGK',
'BGL','BGM','BGN','BGO','BGP','BGQ','BGR','BGS','BGT','BGU','BGV','BGW','BGX','BGY','BGZ','BHA','BHB','BHC','BHD','BHE','BHF','BHG','BHH','BHI','BHJ','BHK',
'BHL','BHM','BHN','BHO','BHP','BHQ','BHR','BHS','BHT','BHU','BHV','BHW','BHX','BHY','BHZ','BIA','BIB','BIC','BID','BIE','BIF','BIG','BIH','BII','BIJ','BIK',
'BIL','BIM','BIN','BIO','BIP','BIQ','BIR','BIS','BIT','BIU','BIV','BIW','BIX','BIY','BIZ','BJA','BJB','BJC','BJD','BJE','BJF','BJG','BJH','BJI','BJJ','BJK',
'BJL','BJM','BJN','BJO','BJP','BJQ','BJR','BJS','BJT','BJU','BJV','BJW','BJX','BJY','BJZ','BKA','BKB','BKC','BKD','BKE','BKF','BKG','BKH','BKI','BKJ','BKK',
'BKL','BKM','BKN','BKO','BKP','BKQ','BKR','BKS','BKT','BKU','BKV','BKW','BKX','BKY','BKZ','BLA','BLB','BLC','BLD','BLE','BLF','BLG','BLH','BLI','BLJ','BLK',
'BLL','BLM','BLN','BLO','BLP','BLQ','BLR','BLS','BLT','BLU','BLV','BLW','BLX','BLY','BLZ','BMA','BMB','BMC','BMD','BME','BMF','BMG','BMH','BMI','BMJ','BMK',
'BML','BMM','BMN','BMO','BMP','BMQ','BMR','BMS','BMT','BMU','BMV','BMW','BMX','BMY','BMZ','BNA','BNB','BNC','BND','BNE','BNF','BNG','BNH','BNI','BNJ','BNK',
'BNL','BNM','BNN','BNO','BNP','BNQ','BNR','BNS','BNT','BNU','BNV','BNW','BNX','BNY','BNZ','BOA','BOB','BOC','BOD','BOE','BOF','BOG','BOH','BOI','BOJ','BOK',
'BOL','BOM','BON','BOO','BOP','BOQ','BOR','BOS','BOT','BOU','BOV','BOW','BOX','BOY','BOZ','BPA','BPB','BPC','BPD','BPE','BPF','BPG','BPH','BPI','BPJ','BPK',
'BPL','BPM','BPN','BPO','BPP','BPQ','BPR','BPS','BPT','BPU','BPV','BPW','BPX','BPY','BPZ','BQA','BQB','BQC','BQD','BQE','BQF','BQG','BQH','BQI','BQJ','BQK',
'BQL','BQM','BQN','BQO','BQP','BQQ','BQR','BQS','BQT','BQU','BQV','BQW','BQX','BQY','BQZ','BRA','BRB','BRC','BRD','BRE','BRF','BRG','BRH','BRI','BRJ','BRK',
'BRL','BRM','BRN','BRO','BRP','BRQ','BRR','BRS','BRT','BRU','BRV','BRW','BRX','BRY','BRZ','BSA','BSB','BSC','BSD','BSE','BSF','BSG','BSH','BSI','BSJ','BSK',
'BSL','BSM','BSN','BSO','BSP','BSQ','BSR','BSS','BST','BSU','BSV','BSW','BSX','BSY','BSZ','BTA','BTB','BTC','BTD','BTE','BTF','BTG','BTH','BTI','BTJ','BTK',
'BTL','BTM','BTN','BTO','BTP','BTQ','BTR','BTS','BTT','BTU','BTV','BTW','BTX','BTY','BTZ','BUA','BUB','BUC','BUD','BUE','BUF','BUG','BUH','BUI','BUJ','BUK',
'BUL','BUM','BUN','BUO','BUP','BUQ','BUR','BUS','BUT','BUU','BUV','BUW','BUX','BUY','BUZ','BVA','BVB','BVC','BVD','BVE','BVF','BVG','BVH','BVI','BVJ','BVK',
'BVL','BVM','BVN','BVO','BVP','BVQ','BVR','BVS','BVT','BVU','BVV','BVW','BVX','BVY','BVZ','BWA','BWB','BWC','BWD','BWE','BWF','BWG','BWH','BWI','BWJ','BWK',
'BWL','BWM','BWN','BWO','BWP','BWQ','BWR','BWS','BWT','BWU','BWV','BWW','BWX','BWY','BWZ','BXA','BXB','BXC','BXD','BXE','BXF','BXG','BXH','BXI','BXJ','BXK',
'BXL','BXM','BXN','BXO','BXP','BXQ','BXR','BXS','BXT','BXU','BXV','BXW','BXX','BXY','BXZ','BYA','BYB','BYC','BYD','BYE','BYF','BYG','BYH','BYI','BYJ','BYK',
'BYL','BYM','BYN','BYO','BYP','BYQ','BYR','BYS','BYT','BYU','BYV','BYW','BYX','BYY','BYZ','BZA','BZB','BZC','BZD','BZE','BZF','BZG','BZH','BZI','BZJ','BZK',
'BZL','BZM','BZN','BZO','BZP','BZQ','BZR','BZS','BZT','BZU','BZV','BZW','BZX','BZY','BZZ','CAA','CAB','CAC','CAD','CAE','CAF','CAG','CAH','CAI','CAJ','CAK',
'CAL','CAM','CAN','CAO','CAP','CAQ','CAR','CAS','CAT','CAU','CAV','CAW','CAX','CAY','CAZ','CBA','CBB','CBC','CBD','CBE','CBF','CBG','CBH','CBI','CBJ','CBK',
'CBL','CBM','CBN','CBO','CBP','CBQ','CBR','CBS','CBT','CBU','CBV','CBW','CBX','CBY','CBZ','CCA','CCB','CCC','CCD','CCE','CCF','CCG','CCH','CCI','CCJ','CCK',
'CCL','CCM','CCN','CCO','CCP','CCQ','CCR','CCS','CCT','CCU','CCV','CCW','CCX','CCY','CCZ','CDA','CDB','CDC','CDD','CDE','CDF','CDG','CDH','CDI','CDJ','CDK',
'CDL','CDM','CDN','CDO','CDP','CDQ','CDR','CDS','CDT','CDU','CDV','CDW','CDX','CDY','CDZ','CEA','CEB','CEC','CED','CEE','CEF','CEG','CEH','CEI','CEJ','CEK',
'CEL','CEM','CEN','CEO','CEP','CEQ','CER','CES','CET','CEU','CEV','CEW','CEX','CEY','CEZ','CFA','CFB','CFC','CFD','CFE','CFF','CFG','CFH','CFI','CFJ','CFK',
'CFL','CFM','CFN','CFO','CFP','CFQ','CFR','CFS','CFT','CFU','CFV','CFW','CFX','CFY','CFZ','CGA','CGB','CGC','CGD','CGE','CGF','CGG','CGH','CGI','CGJ','CGK',
'CGL','CGM','CGN','CGO','CGP','CGQ','CGR','CGS','CGT','CGU','CGV','CGW','CGX','CGY','CGZ','CHA','CHB','CHC','CHD','CHE','CHF','CHG','CHH','CHI','CHJ','CHK',
'CHL','CHM','CHN','CHO','CHP','CHQ','CHR','CHS','CHT','CHU','CHV','CHW','CHX','CHY','CHZ','CIA','CIB','CIC','CID','CIE','CIF','CIG','CIH','CII','CIJ','CIK',
'CIL','CIM','CIN','CIO','CIP','CIQ','CIR','CIS','CIT','CIU','CIV','CIW','CIX','CIY','CIZ','CJA','CJB','CJC','CJD','CJE','CJF','CJG','CJH','CJI','CJJ','CJK',
'CJL','CJM','CJN','CJO','CJP','CJQ','CJR','CJS','CJT','CJU','CJV','CJW','CJX','CJY','CJZ','CKA','CKB','CKC','CKD','CKE','CKF','CKG','CKH','CKI','CKJ','CKK',
'CKL','CKM','CKN','CKO','CKP','CKQ','CKR','CKS','CKT','CKU','CKV','CKW','CKX','CKY','CKZ','CLA','CLB','CLC','CLD','CLE','CLF','CLG','CLH','CLI','CLJ','CLK',
'CLL','CLM','CLN','CLO','CLP','CLQ','CLR','CLS','CLT','CLU','CLV','CLW','CLX','CLY','CLZ','CMA','CMB','CMC','CMD','CME','CMF','CMG','CMH','CMI','CMJ','CMK',
'CML','CMM','CMN','CMO','CMP','CMQ','CMR','CMS','CMT','CMU','CMV','CMW','CMX','CMY','CMZ','CNA','CNB','CNC','CND','CNE','CNF','CNG','CNH','CNI','CNJ','CNK',
'CNL','CNM','CNN','CNO','CNP','CNQ','CNR','CNS','CNT','CNU','CNV','CNW','CNX','CNY','CNZ','COA','COB','COC','COD','COE','COF','COG','COH','COI','COJ','COK',
'COL','COM','CON','COO','COP','COQ','COR','COS','COT','COU','COV','COW','COX','COY','COZ','CPA','CPB','CPC','CPD','CPE','CPF','CPG','CPH','CPI','CPJ','CPK',
'CPL','CPM','CPN','CPO','CPP','CPQ','CPR','CPS','CPT','CPU','CPV','CPW','CPX','CPY','CPZ','CQA','CQB','CQC','CQD','CQE','CQF','CQG','CQH','CQI','CQJ','CQK',
'CQL','CQM','CQN','CQO','CQP','CQQ','CQR','CQS','CQT','CQU','CQV','CQW','CQX','CQY','CQZ','CRA','CRB','CRC','CRD','CRE','CRF','CRG','CRH','CRI','CRJ','CRK',
'CRL','CRM','CRN','CRO','CRP','CRQ','CRR','CRS','CRT','CRU','CRV','CRW','CRX','CRY','CRZ','CSA','CSB','CSC','CSD','CSE','CSF','CSG','CSH','CSI','CSJ','CSK',
'CSL','CSM','CSN','CSO','CSP','CSQ','CSR','CSS','CST','CSU','CSV','CSW','CSX','CSY','CSZ','CTA','CTB','CTC','CTD','CTE','CTF','CTG','CTH','CTI','CTJ','CTK',
'CTL','CTM','CTN','CTO','CTP','CTQ','CTR','CTS','CTT','CTU','CTV','CTW','CTX','CTY','CTZ','CUA','CUB','CUC','CUD','CUE','CUF','CUG','CUH','CUI','CUJ','CUK',
'CUL','CUM','CUN','CUO','CUP','CUQ','CUR','CUS','CUT','CUU','CUV','CUW','CUX','CUY','CUZ','CVA','CVB','CVC','CVD','CVE','CVF','CVG','CVH','CVI','CVJ','CVK',
'CVL','CVM','CVN','CVO','CVP','CVQ','CVR','CVS','CVT','CVU','CVV','CVW','CVX','CVY','CVZ','CWA','CWB','CWC','CWD','CWE','CWF','CWG','CWH','CWI','CWJ','CWK',
'CWL','CWM','CWN','CWO','CWP','CWQ','CWR','CWS','CWT','CWU','CWV','CWW','CWX','CWY','CWZ','CXA','CXB','CXC','CXD','CXE','CXF','CXG','CXH','CXI','CXJ','CXK',
'CXL','CXM','CXN','CXO','CXP','CXQ','CXR','CXS','CXT','CXU','CXV','CXW','CXX','CXY','CXZ','CYA','CYB','CYC','CYD','CYE','CYF','CYG','CYH','CYI','CYJ','CYK',
'CYL','CYM','CYN','CYO','CYP','CYQ','CYR','CYS','CYT','CYU','CYV']

# randomseed = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
randomseed.sort()


i = 0
randomseeded = []
f = open("SampleTest.csv", "w")
while True:
    value = random.choices(randomseed, k=10)

    for tmp in value:
        f.write(tmp+'\n')
        if tmp not in randomseeded:
            randomseeded.append(tmp)
            randomseeded.sort()
            i += 1

    if randomseed == randomseeded:
        break
print(i)
f.close()