# Classify-Exam-Paper-and-Lecture-Handout-by-use-CNN

## เกี่ยวกับโปรเจ็ค

- โปรเจ็คนี้ใช้ Convolutional Neural Network(CNN) มาแยกประเภทเอกสารสองประเภทคือ เอกสารประกอบการสอน และกระดาษข้อสอบที่มีหน้าปกตามแบบฟอร์มที่มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าธนบุรี(มจธ.)กำหนด  โดยสกุลไฟล์ต้องเป็น PDF เท่านั้น
- เราจะทำการแปลงไฟล์ PDF ให้เป็นรูปภาพ แล้วจากนั้นจึงป้อนเข้า CNN Model เพื่อทำการแยกประเภทไฟล์
- ในโปรเจ็คนี้ได้เตรียม dataset สำหรับการ Train และ Test ไว้ที่ 1260 ภาพ โดยแบ่งเป็น Train 1200 ภาพ และ Test 60 ภาพ

![](Readme_image/example_exam(1).jpg =200x200)

### ขั้นตอนการทำงานของโมเดล

1. รับไฟล์ PDF เข้ามาแล้วแปลงหน้าแรกให้เป็นรูปภาพ
2. ย่อขนาดภาพให้เล็กลงเหลือ 150x150x3
3. นำภาพเข้าสู่ CNN[1] ทำการใส่ filter[2] แล้ว convolve เพื่อหา feature ต่างๆจากภาพ
4. Pooling[3] เพื่อดึงค่าจาก bit ที่แข็งแรงของภาพออกมา
5. Flatten bit แล้วป้อนเข้า Neural Network
6. Neural Network ให้ผลเป็นชื่อ class ออกมาว่าเป็นเอกสารประเภทใด

### การทดสอบ

1. นำไฟล์ PDF มาแปลงหน้าแรกให้เป็นรูปภาพโดยใช้ code จากไฟล์ [create_image.py](create_image.py)
2. โหลด weight (100EPOCH.h5) มาใช้กับโมเดลในไฟล์ [model.py](model.py)
3. ใช้ฟังก์ชั่น show_test() เพื่อแสดงผลการทดสอบ
