import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: metalico_naranja_material
        objectName: "Metalico_naranja"
        baseColor: "#ffcc1104"
        indexOfRefraction: 1.4500000476837158
    }

    // Nodes:
    Node {
        id: nao6_Right_Arm_Shoulderpad_obj
        objectName: "Nao6_Right_Arm_Shoulderpad.obj"
        Model {
            id: nao6_Right_Arm_Shoulderpad
            objectName: "Nao6_Right_Arm_Shoulderpad"
            source: "meshes/nao6_Right_Arm_Shoulderpad_mesh.mesh"
            materials: [
                metalico_naranja_material
            ]
        }
    }

    // Animations:
}
