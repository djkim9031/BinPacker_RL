from utils import *
 
xml_data = ET.Element('mujoco')

#Visual field
visual = ET.SubElement(xml_data, 'visual')
headlight = ET.SubElement(visual, 'headlight')
headlight.set('ambient', '0.3 0.3 0.3')

#Asset skybox
asset = ET.SubElement(xml_data, 'asset')
texture = ET.SubElement(asset, 'texture')
texture.set('type', 'skybox')
texture.set('builtin', 'gradient')
texture.set('rgb1', '1.0 0.7 0.4')
texture.set('rgb2', '0.5 0.5 0.5')
texture.set('width', '512')
texture.set('height', '512')

#World creation
worldbody = ET.SubElement(xml_data, 'worldbody')
light = ET.SubElement(worldbody, 'light')
plane1 = ET.SubElement(worldbody, 'geom')
plane2 = ET.SubElement(worldbody, 'geom')

light.set('diffuse', '.25 .25 .25')
light.set('pos', '0 20 20')
light.set('dir', '1 -1 -2')

plane1.set('type', 'plane')
plane2.set('pos', '10 10 0.1')
plane1.set('size', '100 100 0.1')
plane1.set('rgba', '.5 .5 .5 1')

plane2.set('type', 'plane')
plane2.set('pos', '10 10 0.1')
plane2.set('size', '5 5 0.1')
plane2.set('rgba', '.7 .7 .5 .5')


#populate objects
nSKU1 = 5
SKU1_size = '2 1 1'
SKU1_rgba = '.9 .9 0 0.5'
SKU1_mass = '1'

nSKU2 = 4
SKU2_size = '1.5 1 1'
SKU2_rgba = '0 .9 .1 0.5'
SKU2_mass = '3'

nSKU3 = 3
SKU3_size = '1 1 1'
SKU3_rgba = '0 .1 .9 0.5'
SKU3_mass = '2'

nSKU4 = 2
SKU4_size = '2 2 2'
SKU4_rgba = '.9 .1 0 0.5'
SKU4_mass = '5'

create_box_xml(worldbody, nSKU1, SKU1_size, SKU1_rgba, SKU1_mass, '25 15 0.1')
create_box_xml(worldbody, nSKU2, SKU2_size, SKU2_rgba, SKU2_mass, '20 10 0.1')
create_box_xml(worldbody, nSKU3, SKU3_size, SKU3_rgba, SKU3_mass, '20 15 0.1')
create_box_xml(worldbody, nSKU4, SKU4_size, SKU4_rgba, SKU4_mass, '25 10 0.1')

#byte object for flushing
b_xml = ET.tostring(xml_data)
 
#create an xml file
with open("xml/environment.xml", "wb") as f:
    f.write(b_xml)