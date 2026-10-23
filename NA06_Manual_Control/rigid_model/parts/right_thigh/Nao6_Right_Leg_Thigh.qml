import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: plastico_BLANCO_023_material
        objectName: "PLASTICO_BLANCO.023"
        baseColor: "#ffcccccc"
        indexOfRefraction: 1.4500000476837158
    }
    PrincipledMaterial {
        id: altavoz_047_material
        objectName: "ALTAVOZ.047"
        baseColor: "#ff555555"
        indexOfRefraction: 1.4500000476837158
    }
    PrincipledMaterial {
        id: material_024_material
        objectName: "Material.024"
        baseColor: "#ff141414"
        indexOfRefraction: 1.4500000476837158
    }

    // Nodes:
    Node {
        id: nao6_Right_Leg_Thigh_obj
        objectName: "Nao6_Right_Leg_Thigh.obj"
        Model {
            id: nao6_Right_Leg_Thigh
            objectName: "Nao6_Right_Leg_Thigh"
            source: "meshes/nao6_Right_Leg_Thigh_mesh.mesh"
            materials: [
                plastico_BLANCO_023_material,
                altavoz_047_material,
                material_024_material
            ]
        }
    }

    // Animations:
}
