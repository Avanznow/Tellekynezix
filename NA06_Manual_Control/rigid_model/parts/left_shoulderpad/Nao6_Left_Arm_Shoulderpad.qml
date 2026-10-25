import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: metalico_naranja_004_material
        objectName: "Metalico_naranja.004"
        baseColor: "#ffcc1104"
        indexOfRefraction: 1.4500000476837158
    }

    // Nodes:
    Node {
        id: nao6_Left_Arm_Shoulderpad_obj
        objectName: "Nao6_Left_Arm_Shoulderpad.obj"
        Model {
            id: nao6_Left_Arm_Shoulderpad
            objectName: "Nao6_Left_Arm_Shoulderpad"
            source: "meshes/nao6_Left_Arm_Shoulderpad_mesh.mesh"
            materials: [
                metalico_naranja_004_material
            ]
        }
    }

    // Animations:
}
